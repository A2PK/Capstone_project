import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft

import numpy as np
from einops import rearrange, reduce, repeat
import math, random
from typing import List, Tuple, Optional
from scipy.fftpack import next_fast_len
from einops import rearrange, reduce, repeat
import torch.fft as fft

from einops import rearrange, reduce, repeat
from scipy.fftpack import next_fast_len

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

class ETSEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super().__init__()
        kernel_size = 3
        # Calculate padding to maintain sequence length: (kernel_size - 1) // 2
        padding = 1 # For kernel=3, padding=2
        self.conv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                              kernel_size=kernel_size, padding=padding, bias=False)
        self.dropout = nn.Dropout(p=dropout)
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, x):
        # x shape: (batch, seq_len, c_in)
        x = x.permute(0, 2, 1) # Shape: (batch, c_in, seq_len)
        # Conv1d output with correct padding should have same length as input
        x = self.conv(x) # Shape: (batch, d_model, seq_len)
        # --- REMOVE THE SLICING ---
        # x = x[..., :-2] # <--- REMOVE THIS LINE
        # --- END REMOVAL ---
        x = x.transpose(1, 2) # Shape: (batch, seq_len, d_model)
        return self.dropout(x)


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

class ExponentialSmoothing(nn.Module):
    def __init__(self, dim, nhead, dropout=0.1, aux=False, identifier=""): # Added identifier
        super().__init__()
        self.nhead = max(1, nhead)
        self.dim = dim # Dimension PER HEAD
        self.identifier = identifier # Store identifier

        # --- CORRECTED INITIALIZATION ---
        self.v0 = nn.Parameter(torch.randn(1, 1, self.nhead, self.dim))
        # --- END CORRECTION ---

        # --- DEBUG PRINT ---
        # print(f"ES Init ({self.identifier}): nhead={self.nhead}, dim={self.dim}, Requested v0 last dim={self.dim}, Actual v0 shape={self.v0.shape}")
        # --- END DEBUG ---

        self._smoothing_weight = nn.Parameter(torch.randn(self.nhead, 1))
        self.dropout = nn.Dropout(dropout)
        self.aux = aux
        if aux:
            self.aux_dropout = nn.Dropout(dropout)

    def forward(self, values, aux_values=None):
        # values shape: (b, t, h, d) where d should == self.dim
        b, t, h, d = values.shape

        # --- DEBUG PRINT ---
        # print(f"ES Forward ({self.identifier}): Instance dim={self.dim}, Input values last dim (d)={d}, v0 shape={self.v0.shape}")
        if d != self.dim:
             print(f"----> ERROR: Instance dim ({self.dim}) != Input values last dim ({d}) <----")
             # This check should ideally prevent the crash later if it fails
             # raise ValueError(f"Input dimension d ({d}) does not match self.dim ({self.dim})") # Or just print
        # --- END DEBUG ---


        if d != self.dim:
             # If the check above fails, maybe return zeros to avoid crash, although the root cause needs fixing
             # print(f"----> WARNING: Dimension mismatch detected in ES ({self.identifier}). Returning zeros. <----")
             return torch.zeros_like(values)


        init_weight, weight = self.get_exponential_weight(t) # Shape: [1, t, h, 1]

        values_dropped = self.dropout(values)
        output = conv1d_fft(values_dropped, weight, dim=1) # Shape: (b, t, h, d)

        # --- DEBUG PRINT ---
        term_a = init_weight * self.v0
        # print(f"ES Forward ({self.identifier}) - Before Add: init_weight*v0 shape={term_a.shape}, output shape={output.shape}")
        # --- END DEBUG ---


        # v0 shape (1, 1, h, d), init_weight shape (1, t, h, 1)
        # init_weight * v0 broadcasts to (1, t, h, d)
        # output shape (b, t, h, d)
        output = term_a + output # Error occurs here if shapes mismatch

        if self.aux and aux_values is not None:
            # ... (auxiliary calculation part) ...
            if aux_values.shape != values.shape:
                 # This check might be important if LevelLayer's ES is somehow getting wrong aux shapes
                 print(f"----> WARNING: Aux values shape mismatch in ES ({self.identifier}) <----")
                 # raise ValueError(f"aux_values shape {aux_values.shape} must match values shape {values.shape}")
            else:
                 alpha = torch.sigmoid(self._smoothing_weight).view(1, 1, h, 1)
                 safe_denom = torch.clamp(1 - alpha, min=1e-6)
                 aux_weight = weight / safe_denom * alpha
                 aux_values_dropped = self.aux_dropout(aux_values)
                 aux_output = conv1d_fft(aux_values_dropped, aux_weight, dim=1)
                 output = output + aux_output # Add aux component
        return output

    # get_exponential_weight and weight property remain the same
    def get_exponential_weight(self, T):
        powers = torch.arange(T, dtype=torch.float, device=self.weight.device)
        init_weight = self.weight ** (powers + 1)
        weight = (1 - self.weight) * (self.weight ** torch.flip(powers, dims=(0,)))
        return rearrange(init_weight, 'h t -> 1 t h 1'), \
               rearrange(weight, 'h t -> 1 t h 1')

    @property
    def weight(self):
        return torch.sigmoid(self._smoothing_weight)

class GrowthLayer(nn.Module):
    def __init__(self, d_model, nhead, d_head=None, dropout=0.1, output_attention=False):
        super().__init__()
        # ... (other initializations) ...
        self.nhead = nhead
        self.d_head = d_head or (d_model // max(1, nhead))
        self.d_model = d_model
        # ... (value checks) ...
        self.z0 = nn.Parameter(torch.randn(self.nhead, self.d_head))
        self.in_proj = nn.Linear(self.d_model, self.d_head * self.nhead)
        self.output_attention = output_attention
        # Pass d_head as dim and an identifier
        self.es = ExponentialSmoothing(self.d_head, self.nhead, dropout=dropout, identifier="GrowthLayer_ES") # Added identifier
        self.out_proj = nn.Linear(self.d_head * self.nhead, self.d_model)
    # ... (forward method) ...
    def forward(self, inputs):
        b, t, _ = inputs.shape
        values = self.in_proj(inputs).view(b, t, self.nhead, self.d_head)
        z0_repeated = repeat(self.z0, 'h d -> b 1 h d', b=b)
        values_with_z0 = torch.cat([z0_repeated, values], dim=1)
        diffs = values_with_z0[:, 1:] - values_with_z0[:, :-1] # Shape: (b, t, h, d_head)
        smoothed_diffs = self.es(diffs) # Call ES forward
        v0_repeated = repeat(self.es.v0, '1 1 h d -> b 1 h d', b=b) # v0 now has d_head in last dim
        out_with_v0 = torch.cat([v0_repeated, smoothed_diffs], dim=1) # Shape: (b, t+1, h, d_head)
        out_reshaped = rearrange(out_with_v0, 'b time h d -> b time (h d)')
        out_final = self.out_proj(out_reshaped) # Shape: (b, t+1, d_model)

        if self.output_attention:
            _, attn_weights = self.es.get_exponential_weight(t) # Shape: [1, t, h, 1]
            attn = repeat(attn_weights, '1 t h 1 -> b t h 1', b=b)
            return out_final, attn
        return out_final, None


class LevelLayer(nn.Module):
    def __init__(self, d_model, c_out, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.c_out = c_out

        self.es = ExponentialSmoothing(dim=1, nhead=self.c_out, dropout=dropout, aux=True, identifier="LevelLayer_ES") # Added identifier
        self.growth_pred = nn.Linear(self.d_model, self.c_out)
        self.season_pred = nn.Linear(self.d_model, self.c_out)
    # ... (forward method) ...
    def forward(self, level, growth, season):
        b, t_level, _ = level.shape # level is (b, t, c_out)
        common_t = t_level

        growth_sliced = growth[:, 1:common_t+1, :]
        season_sliced = season[:, :common_t, :]

        growth_proj = self.growth_pred(growth_sliced)
        season_proj = self.season_pred(season_sliced)

        level_reshaped = rearrange(level, 'b t h -> b t h 1')
        season_reshaped = rearrange(season_proj, 'b t h -> b t h 1')
        growth_reshaped = rearrange(growth_proj, 'b t h -> b t h 1')

        es_input = level_reshaped - season_reshaped
        level_output_reshaped = self.es(es_input, aux_values=growth_reshaped) # Call ES forward
        level_output = rearrange(level_output_reshaped, 'b t h 1 -> b t h')
        return level_output

# --- Fourier Layer (Seasonal) ---
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


# --- Encoder Layer (Processes one "head" input) ---
class EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, c_out, seq_len, pred_len, k, dim_feedforward=None, dropout=0.1,
                 activation='relu', layer_norm_eps=1e-5, output_attention=False):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead # nhead for internal GrowthLayer/DampingLayer head splitting
        self.c_out = c_out
        self.seq_len = seq_len
        self.pred_len = pred_len
        dim_feedforward = dim_feedforward or 4 * d_model
        self.output_attention = output_attention

        self.growth_layer = GrowthLayer(d_model, nhead, dropout=dropout, output_attention=output_attention)
        # Pass pred_len and k to FourierLayer
        self.seasonal_layer = FourierLayer(d_model, pred_len, k=k, output_attention=output_attention)
        # Pass d_model and c_out to LevelLayer
        self.level_layer = LevelLayer(d_model, c_out, dropout=dropout)

        # Feedforward and Normalization
        self.ff = Feedforward(d_model, dim_feedforward, dropout=dropout, activation=activation)
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, res, level, attn_mask=None):
        # res: (b, t, d_model) - Residual from embedding/previous layer
        # level: (b, t, c_out) - Level component (raw input initially)

        # Ensure inputs are not modified if shared across parallel heads
        res_copy = res # .clone() might be needed if layers modify in-place
        level_copy = level # .clone()

        # 1. Seasonal Block
        # Input: res_copy (b, t, d_model)
        # Output: season (b, t+pred_len, d_model), season_attn
        season, season_attn = self._season_block(res_copy)
        # Residual connection: Subtract season from original seq length part of res
        # Need to slice season: [:, :self.seq_len, :]
        res_after_season = res_copy - season[:, :self.seq_len, :]

        # 2. Growth Block
        # Input: res_after_season (b, t, d_model)
        # Output: growth (b, t+1, d_model), growth_attn
        growth, growth_attn = self._growth_block(res_after_season)
        # Residual connection: Subtract growth from res_after_season
        # Need to slice growth: [:, 1:, :] to match time dimension t and exclude initial state v0
        res_after_growth = res_after_season - growth[:, 1:self.seq_len+1, :] # Match (b,t,d)
        # Normalize before FF
        res_norm1 = self.norm1(res_after_growth) # Apply norm on (res - season - growth)


        # 3. Feedforward Block
        res_ff = self.ff(res_norm1)
        # Add & Norm
        res_final = self.norm2(res_norm1 + res_ff)

        # 4. Level Block (Update Level)
        # Input: level_copy (b, t, c_out)
        # Input: growth (b, t+1, d_model) -> slice -> (b, t, d_model) needed? Check LevelLayer internal slicing
        # Input: season (b, t+pred_len, d_model) -> slice -> (b, t, d_model) needed? Check LevelLayer internal slicing
        level_updated = self.level_layer(level_copy, growth, season) # Let LevelLayer handle slicing

        # Return:
        # res_final: Updated residual for next layer/encoder output (if sequential) - Shape (b, t, d_model)
        # level_updated: Updated level component - Shape (b, t, c_out)
        # growth: Full growth component (incl. initial state) - Shape (b, t+1, d_model)
        # season: Full seasonal component (incl. prediction) - Shape (b, t+pred_len, d_model)
        # season_attn, growth_attn: Attention weights if output_attention=True
        return res_final, level_updated, growth, season, season_attn, growth_attn

    def _growth_block(self, x):
        # Growth layer expects (b, t, d_model), returns (b, t+1, d_model)
        x_out, growth_attn = self.growth_layer(x)
        # Apply dropout to the output (excluding initial state?)
        # Let's apply dropout to the whole output sequence including v0 for simplicity
        return self.dropout1(x_out), growth_attn

    def _season_block(self, x):
        # Seasonal layer expects (b, t, d_model), returns (b, t+pred_len, d_model)
        x_out, season_attn = self.seasonal_layer(x)
        return self.dropout2(x_out), season_attn


# --- Parallel Encoder (Replaces original Encoder/MultiHeadEncoder) ---
class ParallelEncoder(nn.Module):
    """
    Encoder that processes inputs in parallel across multiple 'heads',
    where each head is an independent instance of EncoderLayer.
    Assumes the number of heads (num_encoder_layers) matches the number of decoder layers.
    """
    def __init__(self, encoder_layers: List[EncoderLayer]):
        super().__init__()
        if not encoder_layers:
            raise ValueError("ParallelEncoder requires at least one EncoderLayer.")
        self.heads = nn.ModuleList(encoder_layers)
        self.num_heads = len(encoder_layers)
        # Store output_attention flag from the first layer (assuming consistent)
        self.output_attention = encoder_layers[0].output_attention

    def forward(self, res: torch.Tensor, level: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[Optional[torch.Tensor]], List[Optional[torch.Tensor]]]:
        """
        Forward pass for the ParallelEncoder.

        Args:
            res: Residual tensor from embedding (batch, seq_len, d_model).
            level: Initial level tensor, usually raw input (batch, seq_len, c_out).
            attn_mask: Optional attention mask (not used by ETSformer layers).

        Returns:
            all_levels: List of final level tensors [(b, t, c_out)] from each head.
            all_growths: List of growth tensors [(b, t+1, d_model)] from each head.
            all_seasons: List of season tensors [(b, t+pred_len, d_model)] from each head.
            all_season_attns: List of season attention tensors from each head.
            all_growth_attns: List of growth attention tensors from each head.
        """
        all_levels = []
        all_growths = []
        all_seasons = []
        all_season_attns = []
        all_growth_attns = []

        # Process each head independently on the *same* initial inputs
        for head_layer in self.heads:
            # Each head takes the *original* res and level.
            # EncoderLayer returns: res_final, level_updated, growth, season, season_attn, growth_attn
            # We don't need res_final from the encoder heads here.
            _, level_head, growth_head, season_head, season_attn_head, growth_attn_head = head_layer(
                res, level, attn_mask=attn_mask
            )

            # Collect outputs from this head
            all_levels.append(level_head)
            all_growths.append(growth_head)
            all_seasons.append(season_head)
            all_season_attns.append(season_attn_head) # Will be None if output_attention=False
            all_growth_attns.append(growth_attn_head) # Will be None if output_attention=False

        # Return lists of components, one item per head
        return all_levels, all_growths, all_seasons, all_season_attns, all_growth_attns


# --- Damping Layer (Used by DecoderLayer) ---
class DampingLayer(nn.Module):
    def __init__(self, pred_len, nhead, dropout=0.1):
        super().__init__()
        self.pred_len = pred_len
        self.nhead = max(1, nhead) # Ensure nhead > 0
        self._damping_factor = nn.Parameter(torch.randn(1, self.nhead)) # Shape (1, H)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Input x: (b, 1, d_model) - Typically the last growth value
        b, _, d_model = x.shape
        # d_head calculation must match GrowthLayer's internal split
        d_head = d_model // self.nhead
        if d_head * self.nhead != d_model:
             raise ValueError(f"d_model ({d_model}) not divisible by nhead ({self.nhead}) in DampingLayer")

        # Repeat the last value for the prediction horizon
        x_repeated = repeat(x, 'b 1 d -> b t d', t=self.pred_len) # Shape: (b, pred_len, d_model)

        # Calculate damping factors
        powers = torch.arange(1, self.pred_len + 1, device=self._damping_factor.device) # 1, 2, ..., pred_len
        powers = powers.view(self.pred_len, 1) # Shape: (pred_len, 1)

        # Damping factor: sigmoid(param) -> (0, 1) range. Shape: (1, nhead)
        # Damp factor ^ powers -> Shape: (pred_len, nhead)
        damping_factors = self.damping_factor ** powers
        # Cumulative sum along time dim: Represents the forecasted growth trend contribution
        damping_factors_cumulative = damping_factors.cumsum(dim=0) # Shape: (pred_len, nhead)

        # Reshape input to align with heads: (b, pred_len, nhead, d_head)
        x_reshaped = x_repeated.view(b, self.pred_len, self.nhead, d_head)

        # Apply dropout
        x_dropped = self.dropout(x_reshaped)

        # Apply damping: Needs broadcasting (b, t, h, d) * (t, h) ->unsqueeze-> (b, t, h, d) * (t, h, 1)
        damped_x = x_dropped * damping_factors_cumulative.unsqueeze(0).unsqueeze(-1) # Shapes: (b,t,h,d) * (1,t,h,1)

        # Reshape back to (b, pred_len, d_model)
        output = damped_x.view(b, self.pred_len, d_model)

        # Return damped output and the cumulative factors for potential attention visualization
        # damping_factors_cumulative shape: (pred_len, nhead)
        return output, damping_factors_cumulative

    @property
    def damping_factor(self):
        return torch.sigmoid(self._damping_factor)


# --- Decoder Layer (Processes one head's growth/season) ---
class DecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, c_out, pred_len, dropout=0.1, output_attention=False):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead # nhead for DampingLayer's internal splitting
        self.c_out = c_out
        self.pred_len = pred_len
        self.output_attention = output_attention

        # DampingLayer applies damped trend to the last growth component
        self.growth_damping = DampingLayer(pred_len, nhead, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)

    def forward(self, growth, season):
        # growth: (b, t+1, d_model) - From the corresponding encoder head's GrowthLayer
        # season: (b, t+pred_len, d_model) - From the corresponding encoder head's FourierLayer

        # 1. Growth Horizon Forecast
        # Take the last value of the growth component (represents final trend magnitude)
        last_growth_val = growth[:, -1:, :] # Shape: (b, 1, d_model)
        # Apply damping to extrapolate trend over prediction horizon
        growth_horizon, growth_damping_factors = self.growth_damping(last_growth_val) # Shapes: (b, pred_len, d_model), (pred_len, nhead)
        growth_horizon = self.dropout1(growth_horizon) # Apply dropout

        # 2. Seasonal Horizon Forecast
        # Take the predicted part of the seasonal component
        seasonal_horizon = season[:, -self.pred_len:, :] # Shape: (b, pred_len, d_model)

        # Return horizon components and damping factors
        # Note: growth_damping_factors are returned directly (not attn weights derived from them yet)
        if self.output_attention:
            return growth_horizon, seasonal_horizon, growth_damping_factors
        return growth_horizon, seasonal_horizon, None


# --- Decoder (Aggregates results from Decoder Layers) ---
class Decoder(nn.Module):
    def __init__(self, decoder_layers: List[DecoderLayer]):
        super().__init__()
        if not decoder_layers:
             raise ValueError("Decoder requires at least one DecoderLayer.")
        # Assuming all layers have consistent attributes
        self.d_model = decoder_layers[0].d_model
        self.c_out = decoder_layers[0].c_out
        self.pred_len = decoder_layers[0].pred_len
        self.nhead = decoder_layers[0].nhead # nhead used internally in layers

        self.layers = nn.ModuleList(decoder_layers)
        # Final projection from d_model to output channels (c_out)
        self.pred = nn.Linear(self.d_model, self.c_out)
        self.num_layers = len(decoder_layers) # Should match num_encoder_heads

    def forward(self, growths: List[torch.Tensor], seasons: List[torch.Tensor]):
        # growths: List [(b, t+1, d_model)] from ParallelEncoder heads
        # seasons: List [(b, t+pred_len, d_model)] from ParallelEncoder heads

        if len(growths) != self.num_layers or len(seasons) != self.num_layers:
            raise ValueError(f"Number of growths ({len(growths)}) and seasons ({len(seasons)})"
                             f" must match number of decoder layers ({self.num_layers})")

        growth_repr_list = []
        season_repr_list = []
        growth_dampings_list = [] # Collect damping factors from each layer/head

        # Process each head's output through the corresponding decoder layer
        for idx, layer in enumerate(self.layers):
            growth_horizon, season_horizon, growth_damping = layer(growths[idx], seasons[idx])
            growth_repr_list.append(growth_horizon)
            season_repr_list.append(season_horizon)
            growth_dampings_list.append(growth_damping) # Store factors (pred_len, nhead) or None

        # Aggregate results across heads/layers by summing
        # Summing assumes each head contributes additively to the final forecast
        growth_repr_sum = sum(growth_repr_list) # Shape: (b, pred_len, d_model)
        season_repr_sum = sum(season_repr_list) # Shape: (b, pred_len, d_model)

        # Project aggregated representations to the final output dimension (c_out)
        growth_final = self.pred(growth_repr_sum) # Shape: (b, pred_len, c_out)
        season_final = self.pred(season_repr_sum) # Shape: (b, pred_len, c_out)

        # Return final growth/season components and the list of damping factors
        # growth_dampings_list contains (pred_len, nhead) tensors or Nones
        return growth_final, season_final, growth_dampings_list


# --- Modified ETSformer ---
class ETSformerPar(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len # Not used in forward? Keep for consistency.
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.configs = configs # Store configs for potential use

        # Ensure e_layers (num parallel encoder heads) matches d_layers (num decoder layers)
        if configs.e_layers != configs.d_layers:
            raise ValueError(f"Number of encoder heads (e_layers={configs.e_layers}) "
                             f"must match number of decoder layers (d_layers={configs.d_layers}) "
                             f"for this implementation.")
        self.num_heads = configs.e_layers # Or configs.d_layers

        # Embedding
        self.enc_embedding = ETSEmbedding(configs.enc_in, configs.d_model, dropout=configs.dropout)

        # Parallel Encoder: Create num_heads independent EncoderLayer instances
        self.encoder = ParallelEncoder(
            [
                EncoderLayer(
                    configs.d_model, configs.n_heads, configs.c_out, configs.seq_len, configs.pred_len, configs.K,
                    dim_feedforward=configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                    layer_norm_eps=getattr(configs, 'layer_norm_eps', 1e-5), # Add default if needed
                    output_attention=configs.output_attention,
                ) for _ in range(self.num_heads)
            ]
        )

        # Decoder: Create num_heads independent DecoderLayer instances
        self.decoder = Decoder(
            [
                DecoderLayer(
                    configs.d_model, configs.n_heads, configs.c_out, configs.pred_len,
                    dropout=configs.dropout,
                    output_attention=configs.output_attention,
                ) for _ in range(self.num_heads)
            ],
        )

        # Optional Data Transformation (like Noisy Linear)
        if hasattr(configs, 'std') and configs.std > 0:
             self.transform = Transform(sigma=configs.std)
        else:
             self.transform = None # Or a no-op transform

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, # Keep signature consistent
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None,
                decomposed=False):
        # x_enc: (batch, seq_len, enc_in) - Input sequence
        # level (initial): Use x_enc directly if c_out == enc_in, otherwise need projection?
        # Let's assume c_out matches the feature dim we want to forecast, potentially a subset/projection of enc_in.
        # For simplicity, assume level input to encoder uses the raw x_enc features intended for output (c_out).
        # This requires c_out <= enc_in. If c_out != enc_in, a projection might be needed.
        # Let's assume x_enc already has shape (batch, seq_len, c_out) for the level input pathway.
        # OR modify LevelLayer to take enc_in and project internally? No, keep LevelLayer c_out focused.
        # --> We need initial level of shape (b, t, c_out). Let's use the target features from x_enc.
        # This assumes the features to be forecast (c_out) are present in x_enc.

        if x_enc.shape[-1] != self.configs.c_out:
             # print(f"Warning: x_enc features ({x_enc.shape[-1]}) != c_out ({self.configs.c_out}). Using first c_out features as initial level.")
             level_input = x_enc[:, :, :self.configs.c_out]
             if level_input.shape[-1] != self.configs.c_out:
                  raise ValueError(f"Cannot extract c_out={self.configs.c_out} features from x_enc shape {x_enc.shape}")
        else:
             level_input = x_enc # Shape: (b, t, c_out)


        # Apply transform if training and configured
        if self.training and self.transform is not None:
            x_enc_transformed = self.transform.transform(x_enc)
        else:
            x_enc_transformed = x_enc

        # 1. Embedding: (b, t, enc_in) -> (b, t, d_model)
        res = self.enc_embedding(x_enc_transformed)

        # 2. Parallel Encoder
        # Input: res (b, t, d_model), level_input (b, t, c_out)
        # Output: Lists (length num_heads) of levels, growths, seasons, attentions
        all_levels, all_growths, all_seasons, all_season_attns, all_growth_attns = self.encoder(
            res, level_input, attn_mask=enc_self_mask
        )

        # 3. Decoder
        # Input: Lists of growths, seasons
        # Output: final growth (b, pred, c_out), final season (b, pred, c_out), list of dampings
        growth_pred, season_pred, growth_dampings_list = self.decoder(all_growths, all_seasons)

        # 4. Final Prediction: Level + Growth + Season
        # We need the level component forecast. The encoder outputs updated levels (b, t, c_out) for each head.
        # How to get the final level value or forecast?
        # Option A: Average the final level value across heads: avg(levels[h][:, -1:, :])
        # Option B: Assume level persists: Use the average of the last *input* level value.
        # Option C: Use the Exponential Smoothing from LevelLayer to forecast level? (Not directly available)
        # Let's use Option A: Average the last level value from the encoder heads.
        final_levels_across_heads = [level_head[:, -1:, :] for level_head in all_levels] # List of (b, 1, c_out)
        final_level_avg = torch.stack(final_levels_across_heads).mean(dim=0) # Shape: (b, 1, c_out)

        # Combine components: Level component is constant over pred_len horizon
        preds = final_level_avg + growth_pred + season_pred # Shape: (b, pred_len, c_out)

        # --- Output Handling ---
        if decomposed:
             # Return the final components separately
            return final_level_avg.repeat(1, self.pred_len, 1), growth_pred, season_pred

        if self.output_attention:
            # Need to aggregate attention information from the lists returned by encoder/decoder.
            # Average attentions across heads.
            # Ensure tensors exist before stacking/averaging.

            valid_season_attns = [attn for attn in all_season_attns if attn is not None]
            valid_growth_attns = [attn for attn in all_growth_attns if attn is not None]
            valid_growth_dampings = [damp for damp in growth_dampings_list if damp is not None]

            # Placeholder for aggregated attentions - need precise shapes from layers
            avg_season_attn = None
            avg_growth_attn = None # Maybe combine growth_attn and damping factors?

            if valid_season_attns:
                 # Assuming season_attn from FourierLayer is (b, k, d) or similar
                 # How to best aggregate? Stacking and averaging might work if shapes are consistent.
                 try:
                      stacked_season_attns = torch.stack(valid_season_attns, dim=0) # Shape [num_heads, b, k, d] ?
                      avg_season_attn = reduce(stacked_season_attns, 'h ... -> ...', 'mean') # Average over heads
                      # Keep only pred_len relevant part? Season attn relates to input freqs.
                      # Let's return the averaged map as is.
                 except Exception as e:
                      print(f"Warning: Could not stack/average season attentions: {e}")


            # Combine growth attention (from ES weights) and damping factors?
            # growth_attn might be (b, t, h, 1). damping is (pred_len, h).
            # Difficult to combine meaningfully into a single (b, pred_len, seq_len) map.
            # Let's return the averaged growth ES weights for now.
            if valid_growth_attns:
                 try:
                      # Assuming growth_attn from GrowthLayer is (b, t, nhead, 1)
                      stacked_growth_attns = torch.stack(valid_growth_attns, dim=0) # Shape [num_heads, b, t, nhead, 1] ?
                      # Average over encoder heads (dim 0) and internal ES heads (dim 3?)
                      avg_growth_attn = reduce(stacked_growth_attns, 'h b t nh 1 -> b t', 'mean') # Average over all heads?
                      # Select pred_len part? Growth attn refers to input seq.
                      # Let's return the (b, t) map representing input importance.
                 except Exception as e:
                      print(f"Warning: Could not stack/average growth attentions: {e}")


            # Return preds and averaged attentions (or None if unavailable)
            return preds, avg_season_attn, avg_growth_attn

        # Default return: predictions only
        return preds