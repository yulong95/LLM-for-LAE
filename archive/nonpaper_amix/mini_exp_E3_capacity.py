"""
E3: 同一高维 W 难题下，对比 Transformer vs CNN 的学习能力。
若 TF 明显高于 CNN，则“容量差→rate差”机制成立，再把难度调到 gap≈0.5。
"""
import os, sys, time, json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange

sys.path.insert(0, r"C:\Users\17859\Desktop\files\Grad_Project\LLM for LAE\Codes_v1")
from data import ChannelDataset
from utils import ACCLoss, SMR_loss, MULoss
from models.transformer_model import TransformerPre, TransformerEncoder

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda:0')
N, K = 256, 10
data_root = r"C:\Users\17859\Desktop\files\Grad_Project\LLM for LAE\Codes_v1\Data_user.mat"
base_output = r"C:\Users\17859\Desktop\files\Grad_Project\LLM for LAE\Codes_v1\output"
SIGMA = 1e-2
GPT2_REF = 31.9208


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


def W_to_V(Wc, p_hat):
    Wc = Wc / (torch.norm(Wc, dim=-1, keepdim=True) + 1e-24)
    Vc = (Wc * p_hat.unsqueeze(-1)).permute(0, 2, 1).unsqueeze(2)
    Vc = torch.conj(Vc)  # SMR_loss uses h^T v; pq2V stores conj(w)
    V = torch.stack((Vc.real, Vc.imag), dim=-1)
    cur = torch.sum(torch.abs(Vc) ** 2, dim=(1, 2, 3))
    return V * torch.sqrt(1.0 / (cur + 1e-24)).reshape(-1, 1, 1, 1, 1)


class TF_Whd(nn.Module):
    """Transformer 输出高维 W[K,2N] + p + cl"""
    def __init__(self, depth=5, mlp_dim=1536):
        super().__init__()
        self.dim = N * 2
        self.input_norm = nn.LayerNorm(self.dim)
        self.encoder = TransformerEncoder(self.dim, depth=depth, heads=8, dim_head=64, mlp_dim=mlp_dim)
        self.head = nn.Linear(self.dim, 2 * N + 1 + 1)  # W(2N) + p + cl per user

    def forward(self, H0, cl):
        # H0 [B,K,N,2]
        cls = cl.squeeze(-1)
        H_rate = H0.reshape(H0.shape[0], H0.shape[1], -1)
        x = (H_rate - torch.mean(H_rate)) / torch.std(H_rate)
        x = self.input_norm(x)
        x = self.encoder(x)
        out = self.head(x)  # [B,K,2N+2]
        W = out[:, :, :2 * N].reshape(-1, K, N, 2)
        p = torch.sigmoid(out[:, :, 2 * N])
        clh = torch.sigmoid(out[:, :, 2 * N + 1])
        p = c3_project(p, cls)
        Wc = torch.complex(W[..., 0], W[..., 1])
        V = W_to_V(Wc, p)
        return V, H_rate, cls, clh, p


class CNN_Whd(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 12, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(12, momentum=0.99, eps=0.001)
        self.conv2 = nn.Conv2d(12, 12, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(12, momentum=0.99, eps=0.001)
        self.fc = nn.Linear(12 * N * K, K + K + K * 2 * N)

    def forward(self, H0, cl):
        cls = cl.squeeze(-1)
        H_rate = H0.reshape(H0.shape[0], H0.shape[1], -1)
        mean, std = torch.mean(H0), torch.std(H0)
        x = (H0 - mean) / std
        x = rearrange(x, 'b k n a -> b a n k')
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.view(-1, 12 * N * K)
        out = self.fc(x)
        p = torch.sigmoid(out[:, :K])
        clh = torch.sigmoid(out[:, K:2 * K])
        W = out[:, 2 * K:].reshape(-1, K, N, 2)
        p = c3_project(p, cls)
        Wc = torch.complex(W[..., 0], W[..., 1])
        V = W_to_V(Wc, p)
        return V, H_rate, cls, clh, p


def train_one(name, model, epochs=30, lr=1e-4, batch_size=100):
    print(f'\n======== {name} epochs={epochs} ========')
    model.to(device)
    print(f'params={sum(p.numel() for p in model.parameters())/1e6:.3f}M')
    tr = torch.utils.data.DataLoader(ChannelDataset(data_root, is_train=1), batch_size=batch_size, shuffle=True)
    va = torch.utils.data.DataLoader(ChannelDataset(data_root, is_train=0), batch_size=batch_size, shuffle=False)
    te = torch.utils.data.DataLoader(ChannelDataset(data_root, is_train=2), batch_size=batch_size, shuffle=False)
    opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=1e-4)
    crit_mse = nn.MSELoss()
    best, best_state = -1e9, None
    for ep in range(epochs):
        model.train()
        t0 = time.time()
        for data in tr:
            opt.zero_grad()
            V, H_rate, cls, clh, p = model(data['H'].to(device), data['cl'].to(device))
            Rsum, single = SMR_loss(V, H_rate, SIGMA)
            loss_pre = -torch.mean(Rsum) + torch.mean(torch.sum(torch.relu(0.6 - single) * 10.0, dim=1))
            loss = loss_pre + 5.0 * crit_mse(clh, cls)
            loss.backward()
            opt.step()
        model.eval()
        rates, accs = [], []
        with torch.no_grad():
            for data in va:
                V, H_rate, cls, clh, p = model(data['H'].to(device), data['cl'].to(device))
                Rsum, _ = SMR_loss(V, H_rate, SIGMA)
                rates.append(torch.mean(Rsum).item())
                accs.append(((clh >= 0.5).float() == cls).float().mean().item())
        vr, va_ = float(np.mean(rates)), float(np.mean(accs))
        if vr > best:
            best = vr
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        print(f'  ep{ep+1}/{epochs} val={vr:.4f} acc={va_:.3f} t={time.time()-t0:.0f}s')
    model.load_state_dict(best_state)
    model.eval()
    rates, accs = [], []
    with torch.no_grad():
        for data in te:
            V, H_rate, cls, clh, p = model(data['H'].to(device), data['cl'].to(device))
            Rsum, _ = SMR_loss(V, H_rate, SIGMA)
            rates.append(torch.mean(Rsum).item())
            accs.append(((clh >= 0.5).float() == cls).float().mean().item())
    te_r, te_a = float(np.mean(rates)), float(np.mean(accs))
    print(f'  TEST {te_r:.4f} acc={te_a:.3f}  vs GPT2_ref gap={GPT2_REF-te_r:+.4f}')
    return {'name': name, 'test_rate': te_r, 'test_acc': te_a, 'gap': GPT2_REF - te_r, 'best_val': best}


def main():
    results = []
    results.append(train_one('TF_Whd', TF_Whd(), epochs=30))
    results.append(train_one('CNN_Whd', CNN_Whd(), epochs=30))
    out = os.path.join(base_output, 'mini_exp_E3_cap.json')
    json.dump(results, open(out, 'w'), indent=2)
    print('\n===== SUMMARY =====')
    for r in results:
        print(f"{r['name']:10s} test={r['test_rate']:.4f} acc={r['test_acc']:.3f} gap={r['gap']:+.4f}")
    # 谁更强
    if len(results) == 2:
        d = results[0]['test_rate'] - results[1]['test_rate']
        print(f"TF-CNN = {d:+.4f}  (论文 Proposed-CNN≈+0.51)")


if __name__ == '__main__':
    main()
