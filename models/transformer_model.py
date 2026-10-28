"""
Transformer baseline: seq2seq joint user classification + multi-user near-field precoding.

Paper baseline (3):
  "Transformer，利用Transformer的序列到序列映射能力同时实现用户分类和多用户近场预编码"

Same I/O contract as Gpt2Model:
  forward(H, cl, mean=None, std=None) -> p_hat, lamda_hat, cl_hat
with identical C3 power projection (alpha_N <= gamma).
"""
import torch
import torch.nn as nn
from einops import rearrange


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class TransformerEncoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout=dropout),
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class TransformerPre(nn.Module):
    """
    Pure Transformer baseline for joint classification + precoding.

    Paper Table II target: ~13.15M parameters (all trainable).
    Config: dim=512 (=2N), depth=5, heads=8, dim_head=64, mlp_dim=1536.

    Args:
        gamma: maximum near-field power ratio (alpha_c). Paper Fig.6/8/9: 0.4.
    """

    def __init__(self,
                 Nt=256,
                 K=10,
                 gamma=0.4,
                 depth=5,
                 heads=8,
                 dim_head=64,
                 mlp_dim=1536,
                 dropout=0.0):
        super().__init__()
        self.Nt = Nt
        self.K = K
        self.P_max = 1
        self.gamma = gamma
        self.dim = Nt * 2

        self.input_norm = nn.LayerNorm(self.dim)
        self.encoder = TransformerEncoder(
            self.dim, depth=depth, heads=heads, dim_head=dim_head,
            mlp_dim=mlp_dim, dropout=dropout,
        )
        self.output_layer = nn.Linear(self.dim, 3)
        self.act = nn.Sigmoid()

    def noise(self, H, SNR):
        sigma = 10 ** (-SNR / 10).to(H.device)
        add_noise = torch.sqrt(sigma / 2) * (torch.randn(*H.shape)).to(H.device)
        add_noise = add_noise * torch.sqrt(torch.mean(torch.abs(H) ** 2))
        return H + add_noise

    def forward(self, H=None, cl=None, mean=None, std=None, use_pred_cl_for_c3=False):
        """H: (B, K, 2N) real/imag concatenated channel. cl: (B, K, 1) or (B, K)."""
        if mean is None:
            mean = torch.mean(H)
        if std is None:
            std = torch.std(H)
        x = (H - mean) / std
        x = self.input_norm(x)
        x = self.encoder(x)
        dec_out = self.act(self.output_layer(x))

        cl = cl.squeeze(-1)
        p_hat = dec_out[:, :, 0]
        lamda_hat = dec_out[:, :, 1]
        cl_hat = dec_out[:, :, 2]
        cl_c3 = ((cl_hat >= 0.5).float() if use_pred_cl_for_c3 else cl)

        # --- C3: alpha_N <= gamma (identical to Gpt2Model / CNN_pre) ---
        p_sum = torch.norm(p_hat, p=2, dim=1, keepdim=True) ** 2
        p_normalized = p_hat / torch.sqrt(p_sum + 1e-8)
        temp_label_0 = torch.norm(p_normalized * cl_c3, p=2, dim=1, keepdim=True) ** 2
        mask0 = temp_label_0 < self.gamma

        norm_label_1 = torch.norm(p_hat * cl_c3, p=2, dim=1, keepdim=True) ** 2
        scale_1 = torch.sqrt(self.P_max * self.gamma / (norm_label_1 + 1e-8))
        normalized_label_1 = p_hat * cl_c3 * scale_1
        norm_label_0 = torch.norm(p_hat * (1 - cl_c3), p=2, dim=1, keepdim=True) ** 2
        scale_0 = torch.sqrt(self.P_max * (1 - self.gamma) / (norm_label_0 + 1e-8))
        normalized_label_0 = p_hat * (1 - cl_c3) * scale_0
        p_hat_0 = normalized_label_1 + normalized_label_0
        p_hat_0 = mask0 * p_normalized + (~mask0) * p_hat_0

        lamda_sum = torch.sum(lamda_hat, dim=1, keepdim=True)
        lamda_normalized = lamda_hat / (lamda_sum + 1e-8)
        temp_label_1 = torch.sum(lamda_normalized * cl_c3, dim=1, keepdim=True)
        mask1 = temp_label_1 < self.gamma

        sum_label_1 = torch.sum(lamda_hat * cl_c3, dim=1, keepdim=True)
        scale_p_1 = self.P_max * self.gamma / (sum_label_1 + 1e-8)
        normalized_P_label_1 = lamda_hat * cl_c3 * scale_p_1
        sum_label_0 = torch.sum(lamda_hat * (1 - cl_c3), dim=1, keepdim=True)
        scale_p_0 = self.P_max * (1 - self.gamma) / (sum_label_0 + 1e-8)
        normalized_P_label_0 = lamda_hat * (1 - cl_c3) * scale_p_0
        lamda_hat_0 = normalized_P_label_1 + normalized_P_label_0
        lamda_hat_0 = mask1 * lamda_normalized + (~mask1) * lamda_hat_0

        return p_hat_0, lamda_hat_0, cl_hat
