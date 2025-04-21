class Config:
    def __init__(self, seq_len, pred_len, enc_in, c_out, d_ff, dropout):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in  # Number of input features (NH4N, CODMn, pH, DO)
        self.c_out = c_out    # Number of output features
        self.d_ff = d_ff
        self.dropout = dropout
        self.epochs = 200

class ConfigETSFormer(Config):
    def __init__(self, base_config: Config):
        super().__init__(
            base_config.seq_len,
            base_config.pred_len,
            base_config.enc_in,
            base_config.c_out,
            base_config.d_ff,
            base_config.dropout
        )
        self.label_len = 12
        self.d_model = 64
        self.n_heads = 4
        self.e_layers = 2
        self.d_layers = 2
        self.K = 1
        self.activation = 'gelu'
        self.output_attention = False
        self.std = 0.2    # Standard deviation for data augmentation

class DictConfigModels:
    def __init__(self,Config):
        self.ETSFormerConfig = ConfigETSFormer(Config)