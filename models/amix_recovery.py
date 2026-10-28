"""
A-mix recovery: W = normalize(H A), V_k = conj(p_k * w_k), total power 1.
Must conjugate V to match utils.SMR_loss (uses h^T v, not h^H v).
"""
import torch


def c3_project(p_hat, cl_c3, gamma=0.4, P_max=1.0):
    p_sum = torch.norm(p_hat, p=2, dim=1, keepdim=True) ** 2
    p_normalized = p_hat / torch.sqrt(p_sum + 1e-8)
    temp_label_0 = torch.norm(p_normalized * cl_c3, p=2, dim=1, keepdim=True) ** 2
    mask0 = temp_label_0 < gamma
    norm_label_1 = torch.norm(p_hat * cl_c3, p=2, dim=1, keepdim=True) ** 2
    scale_1 = torch.sqrt(P_max * gamma / (norm_label_1 + 1e-8))
    normalized_label_1 = p_hat * cl_c3 * scale_1
    norm_label_0 = torch.norm(p_hat * (1 - cl_c3), p=2, dim=1, keepdim=True) ** 2
    scale_0 = torch.sqrt(P_max * (1 - gamma) / (norm_label_0 + 1e-8))
    normalized_label_0 = p_hat * (1 - cl_c3) * scale_0
    return mask0 * p_normalized + (~mask0) * (normalized_label_1 + normalized_label_0)


def amix_to_V(H0, A, p):
    """H0 [B,K,N,2] interleaved-like (re,im last), A [B,K,K] complex, p [B,K] -> V [B,N,1,K,2]."""
    Hc = torch.complex(H0[..., 0], H0[..., 1])  # [B,K,N]
    Wc = torch.einsum('bkj,bjn->bkn', A, Hc)
    Wc = Wc / (torch.norm(Wc, dim=-1, keepdim=True) + 1e-24)
    Vc = (Wc * p.unsqueeze(-1)).permute(0, 2, 1).unsqueeze(2)  # [B,N,1,K]
    Vc = torch.conj(Vc)
    V = torch.stack((Vc.real, Vc.imag), dim=-1)
    cur = torch.sum(torch.abs(Vc) ** 2, dim=(1, 2, 3))
    return V * torch.sqrt(1.0 / (cur + 1e-24)).reshape(-1, 1, 1, 1, 1)


def hrate_from_H0(H0):
    """[B,K,N,2] -> [B,K,2N] for SMR_loss / RateCal-style path."""
    return H0.reshape(H0.shape[0], H0.shape[1], -1)


def split_A_p_cl(out, K):
    """out [..., 2K+2] -> A complex [...,K,K] wait per-token [B,K,2K+2] -> A [B,K,K], p, cl.
    Per user token: A_row (K complex = 2K), p, cl.
    """
    Ar = out[..., :K]
    Ai = out[..., K:2 * K]
    A = torch.complex(Ar, Ai)  # [B,K,K]
    p = torch.sigmoid(out[..., 2 * K])
    clh = torch.sigmoid(out[..., 2 * K + 1])
    return A, p, clh
