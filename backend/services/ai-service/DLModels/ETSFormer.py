import torch
import torch.nn as nn
from einops import rearrange, reduce, repeat


class DampingLayer(nn.Module):

    def __init__(self, pred_len, nhead, dropout=0.1, output_attention=False):
        super().__init__()
        self.pred_len = pred_len
        self.nhead = nhead
        self.output_attention = output_attention
        self._damping_factor = nn.Parameter(torch.randn(1, nhead))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = repeat(x, 'b 1 d -> b t d', t=self.pred_len)
        b, t, d = x.shape

        powers = torch.arange(self.pred_len).to(self._damping_factor.device) + 1
        powers = powers.view(self.pred_len, 1)
        damping_factors = self.damping_factor ** powers
        damping_factors = damping_factors.cumsum(dim=0)
        x = x.view(b, t, self.nhead, -1)
        x = self.dropout(x) * damping_factors.unsqueeze(-1)
        x = x.view(b, t, d)
        if self.output_attention:
            return x, damping_factors
        return x, None

    @property
    def damping_factor(self):
        return torch.sigmoid(self._damping_factor)


class DecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, c_out, pred_len, dropout=0.1, output_attention=False):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.c_out = c_out
        self.pred_len = pred_len
        self.output_attention = output_attention

        self.growth_damping = DampingLayer(pred_len, nhead, dropout=dropout, output_attention=output_attention)
        self.dropout1 = nn.Dropout(dropout)

    def forward(self, growth, season):
        growth_horizon, growth_damping = self.growth_damping(growth[:, -1:])
        growth_horizon = self.dropout1(growth_horizon)

        seasonal_horizon = season[:, -self.pred_len:]

        if self.output_attention:
            return growth_horizon, seasonal_horizon, growth_damping
        return growth_horizon, seasonal_horizon, None


class Decoder(nn.Module):

    def __init__(self, layers):
        super().__init__()
        self.d_model = layers[0].d_model
        self.c_out = layers[0].c_out
        self.pred_len = layers[0].pred_len
        self.nhead = layers[0].nhead

        self.layers = nn.ModuleList(layers)
        self.pred = nn.Linear(self.d_model, self.c_out)

    def forward(self, growths, seasons):
        growth_repr = []
        season_repr = []
        growth_dampings = []

        for idx, layer in enumerate(self.layers):
            growth_horizon, season_horizon, growth_damping = layer(growths[idx], seasons[idx])
            growth_repr.append(growth_horizon)
            season_repr.append(season_horizon)
            growth_dampings.append(growth_damping)
        growth_repr = sum(growth_repr)
        season_repr = sum(season_repr)
        return self.pred(growth_repr), self.pred(season_repr), growth_dampings
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft

import numpy as np
from einops import rearrange, reduce, repeat
import math, random


class GrowthLayer(nn.Module):

    def __init__(self, d_model, nhead, d_head=None, dropout=0.1, output_attention=False):
        super().__init__()
        self.d_head = d_head or (d_model // nhead)
        self.d_model = d_model
        self.nhead = nhead
        self.output_attention = output_attention

        self.z0 = nn.Parameter(torch.randn(self.nhead, self.d_head))
        self.in_proj = nn.Linear(self.d_model, self.d_head * self.nhead)
        self.es = ExponentialSmoothing(self.d_head, self.nhead, dropout=dropout)
        self.out_proj = nn.Linear(self.d_head * self.nhead, self.d_model)

        assert self.d_head * self.nhead == self.d_model, "d_model must be divisible by nhead"

    def forward(self, inputs):
        """
        :param inputs: shape: (batch, seq_len, dim)
        :return: shape: (batch, seq_len, dim)
        """
        b, t, d = inputs.shape
        values = self.in_proj(inputs).view(b, t, self.nhead, -1)
        values = torch.cat([repeat(self.z0, 'h d -> b 1 h d', b=b), values], dim=1)
        values = values[:, 1:] - values[:, :-1]
        out = self.es(values)
        out = torch.cat([repeat(self.es.v0, '1 1 h d -> b 1 h d', b=b), out], dim=1)
        out = rearrange(out, 'b t h d -> b t (h d)')
        out = self.out_proj(out)

        if self.output_attention:
            return out, self.es.get_exponential_weight(t)[1]
        return out, None


class FourierLayer(nn.Module):

    def __init__(self, d_model, pred_len, k=None, low_freq=1, output_attention=False):
        super().__init__()
        self.d_model = d_model
        self.pred_len = pred_len
        self.k = k
        self.low_freq = low_freq
        self.output_attention = output_attention

    def forward(self, x):
        """x: (b, t, d)"""
        b, t, d = x.shape

        if self.k is None or self.k >= t // 2:
             # Avoid issues if k is too large or None; use all frequencies
            if t % 2 == 0:
                k_ = t//2 - self.low_freq -1 # Max possible k excluding DC and
            else:
                k_ = t//2 - self.low_freq # Max possible k excluding DC
            k_ = max(1, k_) # Ensure k is at least 1
            # print(f"Warning: k adjusted from {self.k} to {k_} for FourierLayer (t={t}, low_freq={self.low_freq})")
        else:
            k_ = self.k

        # Compute FFT
        x_freq = fft.rfft(x, dim=1)

        # Select frequency range (excluding low_freq components)
        if t % 2 == 0:
            # Exclude DC to low_freq-1, and Nyquist frequency at -1
            x_freq_high = x_freq[:, self.low_freq:-1]
            frequencies = fft.rfftfreq(t)[self.low_freq:-1]
        else:
            # Exclude DC to low_freq-1
            x_freq_high = x_freq[:, self.low_freq:]
            frequencies = fft.rfftfreq(t)[self.low_freq:]

        # Keep top K frequencies
        # Check if k_ exceeds available frequencies
        if k_ > x_freq_high.shape[1]:
             # print(f"Warning: k_ ({k_}) exceeds available high frequencies ({x_freq_high.shape[1]}). Using all available.")
             k_ = x_freq_high.shape[1]

        if k_ <= 0:
             # print(f"Warning: No frequencies to select (k_={k_}). Returning zero tensor.")
             # Return zero tensor with the expected extrapolated shape
             return torch.zeros((b, t + self.pred_len, d), device=x.device), None


        values, indices = torch.topk(x_freq_high.abs(), k_, dim=1, largest=True, sorted=True)

        # Create mask to select top K frequencies from the original x_freq
        mask = torch.zeros_like(x_freq, dtype=torch.bool)
        # Apply indices offset by low_freq to the mask
        # indices shape: (b, k_, d). Need to map back to x_freq indices.
        top_k_indices_in_x_freq = indices + self.low_freq
        # Use scatter or advanced indexing to set mask
        # Need batch and dim indices aligned with top_k_indices_in_x_freq
        batch_indices = torch.arange(b, device=x.device).view(b, 1, 1).expand(-1, k_, d)
        dim_indices = torch.arange(d, device=x.device).view(1, 1, d).expand(b, k_, -1)
        mask.scatter_(1, top_k_indices_in_x_freq, True) # This might not work if indices are not unique across dim d?

        # Simpler masking: Create full index tuple
        # mesh_b, mesh_d = torch.meshgrid(torch.arange(b), torch.arange(d), indexing='ij')
        # index_tuple = (mesh_b.unsqueeze(1), top_k_indices_in_x_freq, mesh_d.unsqueeze(1)) # Might be complex shape-wise
        # Let's try scatter again carefully:
        mask = torch.zeros_like(x_freq, dtype=torch.bool)
        batch_idx = torch.arange(b, device=x.device)[:, None, None]
        dim_idx = torch.arange(d, device=x.device)[None, None, :]
        mask.scatter_(1, top_k_indices_in_x_freq, True) # Scatter True at the top-k indices for each batch/dim


        # Apply mask
        x_freq_filtered = torch.zeros_like(x_freq)
        x_freq_filtered[mask] = x_freq[mask]

        # Select corresponding frequencies
        # frequencies shape: (num_freqs,). Need to select based on top_k_indices_in_x_freq
        # This is complex as indices are per batch/dim.
        # Alternative: Extrapolate using only the selected top-k components directly.

        # --- Extrapolation using selected components ---
        # Get the actual complex values and frequencies for the top K
        # Need to gather these values using the indices.
        x_freq_topk = torch.gather(x_freq, 1, top_k_indices_in_x_freq) # Shape: (b, k_, d)
        # Gather corresponding frequencies
        freq_indices_in_rfftfreq = top_k_indices_in_x_freq # Indices into the rfft output
        # Need the actual frequency values. rfftfreq depends only on t.
        all_rfft_freqs = fft.rfftfreq(t, device=x.device) # Shape: (t//2 + 1,)
        # Gather frequencies using the indices. Indices are (b, k_, d). Need broadcasting.
        topk_freq_values = torch.gather(all_rfft_freqs.view(1, -1, 1).expand(b, -1, d), 1, freq_indices_in_rfftfreq) # Shape: (b, k_, d)

        # Prepare for extrapolation calculation
        x_freq_extrap = torch.cat([x_freq_topk, x_freq_topk.conj()], dim=1) # Shape: (b, 2*k_, d)
        f_extrap = torch.cat([topk_freq_values, -topk_freq_values], dim=1) # Shape: (b, 2*k_, d)

        # Time values for extrapolation (past + future)
        t_val = torch.arange(t + self.pred_len, dtype=torch.float, device=x.device) # Shape: (T_new,)
        t_val = rearrange(t_val, 'time -> 1 1 time 1') # Shape: (1, 1, T_new, 1)

        # Amplitude and Phase
        amp = rearrange(x_freq_extrap.abs() / t, 'b f d -> b f 1 d') # Shape: (b, 2*k_, 1, d)
        phase = rearrange(x_freq_extrap.angle(), 'b f d -> b f 1 d') # Shape: (b, 2*k_, 1, d)

        # Reshape frequencies for broadcasting with time
        f_extrap_reshaped = rearrange(f_extrap, 'b f d -> b f 1 d') # Shape: (b, 2*k_, 1, d)

        # Calculate time domain signal: amp * cos(2*pi*f*t + phase)
        # Broadcasting: (b,2k,1,d) * cos( (b,2k,1,d) * (1,1,T_new,1) + (b,2k,1,d) )
        x_time = amp * torch.cos(2 * math.pi * f_extrap_reshaped * t_val + phase) # Shape: (b, 2*k_, T_new, d)

        # Sum across frequencies
        x_reconstructed = reduce(x_time, 'b f time d -> b time d', 'sum') # Shape: (b, T_new, d)

        # --- Attention Output ---
        # The original implementation's attention was complex. Let's simplify.
        # We can return the indices of the top-k frequencies as a proxy for attention.
        # Or, return the magnitude of the frequencies.
        # Returning the magnitude seems more informative.
        # values shape: (b, k_, d)
        season_attn = values # Use top-k magnitudes as attention proxy

        if self.output_attention:
             # Maybe return magnitudes reshaped or padded? Let's return as is for now.
            return x_reconstructed, season_attn # (b, t+pred_len, d), (b, k, d)
        else:
            return x_reconstructed, None

    def extrapolate(self, x_freq, f, t):
        x_freq = torch.cat([x_freq, x_freq.conj()], dim=1)
        f = torch.cat([f, -f], dim=1)
        t_val = rearrange(torch.arange(t + self.pred_len, dtype=torch.float),
                      't -> () () t ()').to(x_freq.device)

        amp = rearrange(x_freq.abs() / t, 'b f d -> b f () d')
        phase = rearrange(x_freq.angle(), 'b f d -> b f () d')

        x_time = amp * torch.cos(2 * math.pi * f * t_val + phase)

        return reduce(x_time, 'b f t d -> b t d', 'sum')

    def topk_freq(self, x_freq):
        values, indices = torch.topk(x_freq.abs(), self.k, dim=1, largest=True, sorted=True)
        mesh_a, mesh_b = torch.meshgrid(torch.arange(x_freq.size(0)), torch.arange(x_freq.size(2)))
        index_tuple = (mesh_a.unsqueeze(1), indices, mesh_b.unsqueeze(1))
        x_freq = x_freq[index_tuple]

        return x_freq, index_tuple

    def dft_forward(self, x):
        T = x.size(1)

        dft_mat = fft.fft(torch.eye(T))
        i, j = torch.meshgrid(torch.arange(self.pred_len + T), torch.arange(T))
        omega = np.exp(2 * math.pi * 1j / T)
        idft_mat = (np.power(omega, i * j) / T).cfloat()

        x_freq = torch.einsum('ft,btd->bfd', [dft_mat, x.cfloat()])

        if T % 2 == 0:
            x_freq = x_freq[:, self.low_freq:T // 2]
        else:
            x_freq = x_freq[:, self.low_freq:T // 2 + 1]

        _, indices = torch.topk(x_freq.abs(), self.k, dim=1, largest=True, sorted=True)
        indices = indices + self.low_freq
        indices = torch.cat([indices, -indices], dim=1)

        dft_mat = repeat(dft_mat, 'f t -> b f t d', b=x.shape[0], d=x.shape[-1])
        idft_mat = repeat(idft_mat, 't f -> b t f d', b=x.shape[0], d=x.shape[-1])

        mesh_a, mesh_b = torch.meshgrid(torch.arange(x.size(0)), torch.arange(x.size(2)))

        dft_mask = torch.zeros_like(dft_mat)
        dft_mask[mesh_a, indices, :, mesh_b] = 1
        dft_mat = dft_mat * dft_mask

        idft_mask = torch.zeros_like(idft_mat)
        idft_mask[mesh_a, :, indices, mesh_b] = 1
        idft_mat = idft_mat * idft_mask

        attn = torch.einsum('bofd,bftd->botd', [idft_mat, dft_mat]).real
        return torch.einsum('botd,btd->bod', [attn, x]), rearrange(attn, 'b o t d -> b d o t')


class LevelLayer(nn.Module):

    def __init__(self, d_model, c_out, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.c_out = c_out

        self.es = ExponentialSmoothing(1, self.c_out, dropout=dropout, aux=True)
        self.growth_pred = nn.Linear(self.d_model, self.c_out)
        self.season_pred = nn.Linear(self.d_model, self.c_out)

    def forward(self, level, growth, season):
        
        b, t, _ = level.shape
        growth = self.growth_pred(growth).view(b, t, self.c_out, 1)
        season = self.season_pred(season).view(b, t, self.c_out, 1)
        growth = growth.view(b, t, self.c_out, 1)
        season = season.view(b, t, self.c_out, 1)
        level = level.view(b, t, self.c_out, 1)
        out = self.es(level - season, aux_values=growth)
        out = rearrange(out, 'b t h d -> b t (h d)')
        return out

class EncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, c_out, seq_len, pred_len, k, dim_feedforward=None, dropout=0.1,
                 activation='sigmoid', layer_norm_eps=1e-5, output_attention=False):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.c_out = c_out
        self.seq_len = seq_len
        self.pred_len = pred_len
        dim_feedforward = dim_feedforward or 4 * d_model
        self.dim_feedforward = dim_feedforward

        self.growth_layer = GrowthLayer(d_model, nhead, dropout=dropout, output_attention=output_attention)
        self.seasonal_layer = FourierLayer(d_model, pred_len, k=k, output_attention=output_attention)
        self.level_layer = LevelLayer(d_model, c_out, dropout=dropout)

        # Implementation of Feedforward model
        self.ff = Feedforward(d_model, dim_feedforward, dropout=dropout, activation=activation)
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, res, level, attn_mask=None):
        
        season, season_attn = self._season_block(res)
        res = res - season[:, :-self.pred_len]
        growth, growth_attn = self._growth_block(res)
        
        res = self.norm1(res - growth[:, 1:])
        res = self.norm2(res + self.ff(res))
        
        level = self.level_layer(level, growth[:, :-1], season[:, :-self.pred_len])
        

        return res, level, growth, season, season_attn, growth_attn

    def _growth_block(self, x):
        x, growth_attn = self.growth_layer(x)
        return self.dropout1(x), growth_attn

    def _season_block(self, x):
        x, season_attn = self.seasonal_layer(x)
        return self.dropout2(x), season_attn


class Encoder(nn.Module):

    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, res, level, attn_mask=None):
        growths = []
        seasons = []
        season_attns = []
        growth_attns = []
        for layer in self.layers:
            res, level, growth, season, season_attn, growth_attn = layer(res, level, attn_mask=None)
            growths.append(growth)
            seasons.append(season)
            season_attns.append(season_attn)
            growth_attns.append(growth_attn)

        return level, growths, seasons, season_attns, growth_attns
        import math

import torch
import torch.nn as nn
import torch.fft as fft

from einops import rearrange, reduce, repeat
from scipy.fftpack import next_fast_len


def conv1d_fft(f, g, dim=-1):
    N = f.size(dim)
    M = g.size(dim)

    fast_len = next_fast_len(N + M - 1)

    F_f = fft.rfft(f, fast_len, dim=dim)
    F_g = fft.rfft(g, fast_len, dim=dim)

    F_fg = F_f * F_g.conj()
    out = fft.irfft(F_fg, fast_len, dim=dim)
    out = out.roll((-1,), dims=(dim,))
    idx = torch.as_tensor(range(fast_len - N, fast_len)).to(out.device)
    out = out.index_select(dim, idx)

    return out


class ExponentialSmoothing(nn.Module):

    def __init__(self, dim, nhead, dropout=0.1, aux=False):
        super().__init__()
        self._smoothing_weight = nn.Parameter(torch.randn(nhead, 1))
        self.v0 = nn.Parameter(torch.randn(1, 1, nhead, dim))
        self.dropout = nn.Dropout(dropout)
        if aux:
            self.aux_dropout = nn.Dropout(dropout)

    def forward(self, values, aux_values=None):
        b, t, h, d = values.shape

        init_weight, weight = self.get_exponential_weight(t)
        output = conv1d_fft(self.dropout(values), weight, dim=1)
        output = init_weight * self.v0 + output

        if aux_values is not None:
            aux_weight = weight / (1 - self.weight) * self.weight
            aux_output = conv1d_fft(self.aux_dropout(aux_values), aux_weight)
            output = output + aux_output

        return output

    def get_exponential_weight(self, T):
        # Generate array [0, 1, ..., T-1]
        powers = torch.arange(T, dtype=torch.float, device=self.weight.device)

        # (1 - \alpha) * \alpha^t, for all t = T-1, T-2, ..., 0]
        weight = (1 - self.weight) * (self.weight ** torch.flip(powers, dims=(0,)))

        # \alpha^t for all t = 1, 2, ..., T
        init_weight = self.weight ** (powers + 1)

        return rearrange(init_weight, 'h t -> 1 t h 1'), \
               rearrange(weight, 'h t -> 1 t h 1')

    @property
    def weight(self):
        return torch.sigmoid(self._smoothing_weight)
import torch
import torch.nn as nn
from einops import reduce

class Transform:
    def __init__(self, sigma):
        self.sigma = sigma

    @torch.no_grad()
    def transform(self, x):
        return self.jitter(self.shift(self.scale(x)))

    def jitter(self, x):
        return x + (torch.randn(x.shape).to(x.device) * self.sigma)

    def scale(self, x):
        return x * (torch.randn(x.size(-1)).to(x.device) * self.sigma + 1)

    def shift(self, x):
        return x + (torch.randn(x.size(-1)).to(x.device) * self.sigma)


class ETSformer(nn.Module):

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len

        self.configs = configs

        assert configs.e_layers == configs.d_layers, "Encoder and decoder layers must be equal"

        # Embedding
        self.enc_embedding = ETSEmbedding(configs.enc_in, configs.d_model, dropout=configs.dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    configs.d_model, configs.n_heads, configs.c_out, configs.seq_len, configs.pred_len, configs.K,
                    dim_feedforward=configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                    output_attention=configs.output_attention,
                ) for _ in range(configs.e_layers)
            ]
        )

        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    configs.d_model, configs.n_heads, configs.c_out, configs.pred_len,
                    dropout=configs.dropout,
                    output_attention=configs.output_attention,
                ) for _ in range(configs.d_layers)
            ],
        )

        self.transform = Transform(sigma=self.configs.std)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None,
                decomposed=False, attention=False):
        
        with torch.no_grad():
            if self.training:
                x_enc = self.transform.transform(x_enc)
                
        
        res = self.enc_embedding(x_enc)
        
        level, growths, seasons, season_attns, growth_attns = self.encoder(res, x_enc, attn_mask=enc_self_mask)
        
        growth, season, growth_dampings = self.decoder(growths, seasons)

        if decomposed:
            return level[:, -1:], growth, season

        preds = level[:, -1:] + growth + season

        if attention:
            decoder_growth_attns = []
            for growth_attn, growth_damping in zip(growth_attns, growth_dampings):
                decoder_growth_attns.append(torch.einsum('bth,oh->bhot', [growth_attn.squeeze(-1), growth_damping]))

            season_attns = torch.stack(season_attns, dim=0)[:, :, -self.pred_len:]
            season_attns = reduce(season_attns, 'l b d o t -> b o t', reduction='mean')
            decoder_growth_attns = torch.stack(decoder_growth_attns, dim=0)[:, :, -self.pred_len:]
            decoder_growth_attns = reduce(decoder_growth_attns, 'l b d o t -> b o t', reduction='mean')
            return preds, season_attns, decoder_growth_attns
        return preds
    
import torch.nn as nn
import torch.nn.functional as F


class ETSEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                              kernel_size=3, padding=2, bias=False)
        self.dropout = nn.Dropout(p=dropout)
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, x,):
        x = self.conv(x.permute(0,2,1))[..., :-2]
        return self.dropout(x.transpose(1,2))


class Feedforward(nn.Module):
    def __init__(self, d_model, dim_feedforward, dropout=0.1, activation='sigmoid'):
        # Implementation of Feedforward model
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=False)
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=False)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = getattr(F, activation)

    def forward(self, x):
        x = self.linear2(self.dropout1(self.activation(self.linear1(x))))
        return self.dropout2(x)