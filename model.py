import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================
# 1. Faithful Positional Encoding (FE)
# ==========================================
class FaithfulEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        K = d_model // 2 - 1
        w = torch.zeros(K)
        for k in range(1, K + 1):
            w[k - 1] = 2 * math.pi * k / d_model

        pos = torch.arange(0, max_len).float().unsqueeze(1)
        w = w.unsqueeze(0)

        pe = torch.zeros(max_len, d_model)
        pe[:, 0] = 1.0 / math.sqrt(2)
        pe[:, 1 : 2 * K : 2] = torch.cos(pos * w)
        pe[:, 2 : 2 * K + 1 : 2] = torch.sin(pos * w)
        pe[:, -1] = torch.cos(pos.squeeze(1) * math.pi) / math.sqrt(2)

        pe = pe * math.sqrt(2.0 / d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.size(1)].unsqueeze(0)


class SinusoidalPositionalEncoding(nn.Module):
    """Standard Transformer sinusoidal positional encoding."""

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.size(1)].unsqueeze(0)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.1):
        super().__init__()
        if dim % heads != 0:
            raise ValueError("dim must be divisible by heads")

        self.heads = heads
        self.dim_head = dim // heads
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: t.reshape(B, N, self.heads, self.dim_head).transpose(1, 2),
            qkv,
        )

        q = F.elu(q) + 1.0
        k = F.elu(k) + 1.0

        kv = torch.einsum("...nd,...ne->...de", k, v)
        k_sum = torch.einsum("...nd->...d", k)

        out = torch.einsum("...nd,...de->...ne", q, kv)
        normalizer = torch.einsum("...nd,...d->...n", q, k_sum).unsqueeze(-1)
        out = out / (normalizer + 1e-6)

        out = out.transpose(1, 2).reshape(B, N, C)
        return self.dropout(self.to_out(out))


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.1, use_linear_attn=True):
        super().__init__()
        self.use_linear_attn = use_linear_attn
        self.norm1 = nn.LayerNorm(dim)
        if use_linear_attn:
            self.attn = LinearAttention(dim, heads, dropout)
        else:
            self.attn = nn.MultiheadAttention(
                dim, heads, dropout=dropout, batch_first=True
            )
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, x):
        if self.use_linear_attn:
            x = x + self.attn(self.norm1(x))
        else:
            normed = self.norm1(x)
            attn_out, _ = self.attn(normed, normed, normed, need_weights=False)
            x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


# ==========================================
# 3. Main Model: LST-TFDN (Multivariate CNN Front-End)
# ==========================================
class LST_TFDN(nn.Module):
    def __init__(
        self,
        num_sensors,
        window_size=100,
        d_model=64,
        heads=4,
        dropout=0.2,
        use_linear_attn=True,
        use_fe=True,
        use_cnn=True,
        use_transformer=True,
        num_layers=2,
    ):
        super().__init__()
        self.num_sensors = num_sensors
        self.use_cnn = use_cnn
        self.use_transformer = use_transformer

        if use_cnn:
            self.cnn = nn.Sequential(
                nn.Conv1d(num_sensors, d_model, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm1d(d_model),
                nn.ReLU(),
                nn.Conv1d(d_model, d_model, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm1d(d_model),
                nn.ReLU(),
            )
            self.input_proj = None
            seq_len = window_size // 2 + 1
        else:
            self.cnn = None
            self.input_proj = nn.Linear(num_sensors, d_model)
            seq_len = window_size

        if use_fe:
            self.pos_enc = FaithfulEncoding(d_model, max_len=seq_len)
        else:
            self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=seq_len)

        if use_transformer:
            self.transformer = nn.Sequential(
                *[
                    TransformerBlock(
                        d_model, heads, dropout, use_linear_attn=use_linear_attn
                    )
                    for _ in range(num_layers)
                ]
            )
        else:
            self.transformer = nn.Identity()

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x):
        if self.use_cnn:
            x = self.cnn(x)
            x = x.transpose(1, 2)
        else:
            x = x.transpose(1, 2)
            x = self.input_proj(x)

        x = self.pos_enc(x)
        x = self.transformer(x)
        x = x.transpose(1, 2)
        x = self.pool(x).squeeze(-1)
        out = self.classifier(x)
        return out.squeeze(-1)


def build_standard_transformer_baseline(num_sensors, window_size=100, target_params=1_000_000):
    """
    Standard softmax-attention Transformer baseline (~1M parameters).
    Same CNN front-end and training protocol as LST-TFDN for like-for-like comparison.
    """
    configs = [
        (160, 8, 4),
        (128, 8, 5),
        (128, 8, 4),
        (96, 8, 6),
        (96, 8, 4),
    ]
    best = None
    for d_model, heads, num_layers in configs:
        if d_model % heads != 0:
            continue
        model = LST_TFDN(
            num_sensors=num_sensors,
            window_size=window_size,
            d_model=d_model,
            heads=heads,
            dropout=0.2,
            use_linear_attn=False,
            use_fe=False,
            use_cnn=True,
            use_transformer=True,
            num_layers=num_layers,
        )
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if best is None or abs(n_params - target_params) < abs(best[0] - target_params):
            best = (n_params, d_model, heads, num_layers)

    _, d_model, heads, num_layers = best
    return LST_TFDN(
        num_sensors=num_sensors,
        window_size=window_size,
        d_model=d_model,
        heads=heads,
        dropout=0.2,
        use_linear_attn=False,
        use_fe=False,
        use_cnn=True,
        use_transformer=True,
        num_layers=num_layers,
    )


def build_cnn_only_baseline(num_sensors, window_size=100, d_model=32):
    """CNN front-end + classifier without Transformer stack."""
    return LST_TFDN(
        num_sensors=num_sensors,
        window_size=window_size,
        d_model=d_model,
        heads=4,
        dropout=0.2,
        use_linear_attn=True,
        use_fe=True,
        use_cnn=True,
        use_transformer=False,
    )


def build_lst_tfdn_baseline(num_sensors, window_size=100, d_model=32):
    """Full proposed LST-TFDN configuration."""
    return LST_TFDN(
        num_sensors=num_sensors,
        window_size=window_size,
        d_model=d_model,
        heads=4,
        dropout=0.2,
        use_linear_attn=True,
        use_fe=True,
        use_cnn=True,
        use_transformer=True,
        num_layers=2,
    )


if __name__ == "__main__":
    print("Testing LST-TFDN Model...")

    dummy_input = torch.randn(32, 55, 100)
    model = LST_TFDN(num_sensors=55, window_size=100, d_model=64, heads=4, dropout=0.2)

    output = model(dummy_input)

    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {total_params:,}")
    print("Model test successful!")
