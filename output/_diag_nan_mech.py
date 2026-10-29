"""探针：λ_k=0 时官方 pq2V 中 V_temp_u{k} 变 NaN 的算术机制。只读。"""
import os, sys
import torch
import numpy as np
from einops import rearrange

sys.path.insert(0, r"C:\Users\17859\Desktop\files\Grad_Project\LLM for LAE\Codes_v1")
from data import ChannelDataset
from models.baseline_CNN import CNN_pre
from utils import pq2V

sys.stdout.reconfigure(encoding='utf-8')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda:0')
N, K = 256, 10
SIGMA = 1e-2
data_root = r"C:\Users\17859\Desktop\files\Grad_Project\LLM for LAE\Codes_v1\Data_user.mat"
torch.manual_seed(0)
np.random.seed(0)

Z = torch.load(r"C:\Users\17859\Desktop\files\Grad_Project\LLM for LAE\Codes_v1\output\_diag_zstar_feas.pt",
               map_location='cpu', weights_only=True)
tr_all = ChannelDataset(data_root, is_train=1)
H = torch.stack([tr_all[i]['H'] for i in range(150)]).to(device)
cl = torch.stack([tr_all[i]['cl'] for i in range(150)]).to(device)
H_re = rearrange(H, 'n k W H -> n H W k', H=2)
H0z = torch.zeros(H_re.shape, device=device)
H0z[:, :, :, :K] = H_re[:, :, :, :K]
Hr = rearrange(H0z, 'n H W k -> n k (W H)')

ps = Z['p_star_tgt'].to(device)
ls = Z['l_star_tgt'].to(device)
p_pos = torch.clamp(ps, min=0.0)
l_pos = torch.clamp(ls, min=0.0)

# 首个 NaN 样本 idx=15，λ 的 u7=0
idx, u = 15, 7
p1 = p_pos[idx:idx + 1]
l1 = l_pos[idx:idx + 1]
H1 = Hr[idx:idx + 1]
print(f'样本 idx={idx}  用户 u={u}')
print(f'  p = {[f"{v:.4f}" for v in p1[0].tolist()]}')
print(f'  λ = {[f"{v:.6f}" for v in l1[0].tolist()]}')
print(f'  λ_u{u} == 0 ? {bool(l1[0, u] == 0)}   p_u{u} = {p1[0, u].item()}')

muchannel = rearrange(H1, 'b K (N a c) -> b N c a K', a=2, c=1)
channel = torch.view_as_complex(muchannel.permute(0, 1, 2, 4, 3).contiguous())
p_list = p1.type_as(channel)
q_list = l1.type_as(channel)
Nt = N

q_sqrt = torch.sqrt(q_list)
print(f'\n  q_u{u} = {q_list[0, u].item()}   sqrt(q_u{u}) = {q_sqrt[0, u].item()}')
weighted_P = channel[:, :, 0, :] * q_sqrt.reshape((-1, 1, K)).repeat(1, Nt, 1)
print(f'  |weighted_P[:, :, u]| max = {weighted_P[:, :, u].abs().max().item():.6e}')
B = SIGMA * torch.eye(K).type_as(channel).reshape((1, K, K)).repeat(1, 1, 1)
B = B + torch.matmul(torch.conj(weighted_P.permute(0, 2, 1)), weighted_P)
B_inv = torch.linalg.inv(B)
print(f'  |inv(B)| max = {B_inv.abs().max().item():.6e}')
P_mat = torch.matmul(weighted_P, B_inv)
print(f'  |P_mat[:, :, u]| max = {P_mat[:, :, u].abs().max().item():.6e}')

den = torch.sqrt(q_list[:, u] + 1e-24)
V_pre = P_mat[:, :, u] / den.reshape((-1, 1)).repeat(1, Nt)
print(f'\n  分母 sqrt(λ_u+1e-24) = {den.item()}')
print(f'  V_pre = P_mat/den: |max| = {V_pre.abs().max().item():.6e}  有限 = {bool(torch.isfinite(V_pre.real).all())}')

nrm_raw = torch.norm(V_pre, dim=1)
print(f'  torch.norm(V_pre, dim=1) = {nrm_raw.item()}   dtype={nrm_raw.dtype}  有限 = {bool(torch.isfinite(nrm_raw.real).all() if torch.is_complex(nrm_raw) else torch.isfinite(nrm_raw).all())}')
nrm = nrm_raw.reshape(-1, 1).repeat(1, Nt) + 1e-48
num = (p_list[:, u]).reshape((-1, 1)).repeat(1, Nt) * V_pre
print(f'  p_u * V_pre: |max| = {num.abs().max().item():.6e}')
V_post = num / nrm
print(f'  V_post = num/nrm: |max| = {V_post.abs().max().item():.6e}  有限 = {bool(torch.isfinite(V_post.real).all() and torch.isfinite(V_post.imag).all())}')
bad = (~torch.isfinite(V_post.real) | ~torch.isfinite(V_post.imag)).nonzero(as_tuple=False)
if len(bad):
    i = bad[0].tolist()
    print(f'  首个 NaN 位置 {i}: num={num[tuple(i)]}  nrm={nrm[tuple(i)]}  → num/nrm 的中间量:')
    print(f'    |num| = {num[tuple(i)].abs().item():.6e}  |nrm| = {nrm[tuple(i)].abs().item() if torch.is_complex(nrm) else nrm[tuple(i)].item():.6e}')
    print(f'    若 |nrm| 溢出为 inf 或 num 为 0×inf → NaN；若 nrm 有限而 num 有限则应有限')

# 对照：λ_u = 1e-3（投影下界）时同一用户是否有限
print('\n对照 λ_u=1e-3（LAM_FLOOR）:')
l2 = l1.clone()
l2[0, u] = 1e-3
with torch.no_grad():
    V2 = pq2V(p1, l2, H1, SIGMA, N)
print(f'  官方 pq2V 有限 = {bool(torch.isfinite(V2).all())}')
