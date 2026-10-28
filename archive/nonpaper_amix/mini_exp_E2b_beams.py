"""
E2b: 中等难度波束参数化，找“掉 0.3~0.8”的甜点。
  whd_long — 全 W，40 epoch（看收敛到哪）
  wmed     — 每用户波束 = H 的线性组合 (K 系数) + DFT 窗口 (M 系数)，中等维
  wres     — w_k = normalize(h_k + 低秩残差)
"""
import os, sys, time, json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange

sys.path.insert(0, r"C:\Users\17859\Desktop\files\Grad_Project\LLM for LAE\Codes_v1")
from data import ChannelDataset
from utils import ACCLoss, RateCal, MULoss, SMR_loss
from models.baseline_CNN import CNN_pre

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda:0')
N, K = 256, 10
data_root = r"C:\Users\17859\Desktop\files\Grad_Project\LLM for LAE\Codes_v1\Data_user.mat"
base_output = r"C:\Users\17859\Desktop\files\Grad_Project\LLM for LAE\Codes_v1\output"
GPT2_REF = 31.9208
SIGMA = 1e-2


class Trunk(nn.Module):
    def __init__(self, ch=12):
        super().__init__()
        self.conv1 = nn.Conv2d(2, ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(ch, momentum=0.99, eps=0.001)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(ch, momentum=0.99, eps=0.001)
        self.ch = ch

    def forward(self, x, N, K):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x.view(-1, self.ch * N * K)


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
    """Wc [B,K,N] complex unit-norm columns; p_hat [B,K]; return V [B,N,1,K,2] power 1."""
    Wc = Wc / (torch.norm(Wc, dim=-1, keepdim=True) + 1e-24)
    Vc = (Wc * p_hat.unsqueeze(-1)).permute(0, 2, 1).unsqueeze(2)
    V = torch.stack((Vc.real, Vc.imag), dim=-1)
    cur = torch.sum(torch.abs(Vc) ** 2, dim=(1, 2, 3))
    return V * torch.sqrt(1.0 / (cur + 1e-24)).reshape(-1, 1, 1, 1, 1)


def prep_H(H0):
    """H0 [B,K,N,2] -> Hc [B,K,N] complex, H_rate [B,K,2N] interleaved, mean/std of real tensor."""
    Hc = torch.complex(H0[..., 0], H0[..., 1])
    H_rate = H0.reshape(H0.shape[0], H0.shape[1], -1)  # [B,K,2N] interleaved re/im
    mean = torch.mean(H0)
    std = torch.std(H0)
    return Hc, H_rate, mean, std


class CNN_Whd(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = Trunk(12)
        self.fc = nn.Linear(12 * N * K, K + K + K * 2 * N)

    def forward(self, H0, cl):
        cls = cl.squeeze(-1)
        Hc, H_rate, mean, std = prep_H(H0)
        x = (H0 - mean) / std
        x = rearrange(x, 'b k n a -> b a n k')
        feat = self.trunk(x, N, K)
        out = self.fc(feat)
        p = torch.sigmoid(out[:, :K])
        clh = torch.sigmoid(out[:, K:2 * K])
        W = out[:, 2 * K:].reshape(-1, K, N, 2)
        p = c3_project(p, cls)
        Wc = torch.complex(W[..., 0], W[..., 1])
        V = W_to_V(Wc, p)
        return V, H_rate, cls, clh, p


class CNN_Wmed(nn.Module):
    """波束 = H 的线性组合 + 局部 DFT 窗口组合（中等维）"""
    def __init__(self, M=16):
        super().__init__()
        self.M = M
        self.trunk = Trunk(12)
        self.fc = nn.Linear(12 * N * K, K * K + K * M + K + K)

    def forward(self, H0, cl):
        cls = cl.squeeze(-1)
        Hc, H_rate, mean, std = prep_H(H0)
        x = (H0 - mean) / std
        x = rearrange(x, 'b k n a -> b a n k')
        feat = self.trunk(x, N, K)
        out = self.fc(feat)
        off = 0
        alpha = out[:, off:off + K * K].reshape(-1, K, K); off += K * K
        beta = out[:, off:off + K * self.M].reshape(-1, K, self.M); off += K * self.M
        p = torch.sigmoid(out[:, off:off + K]); off += K
        clh = torch.sigmoid(out[:, off:off + K])
        p = c3_project(p, cls)

        # beam k = sum_j alpha[k,j] * h_j
        mixH = torch.einsum('bkj,bjn->bkn', alpha.to(Hc.dtype), Hc)

        Hf = torch.fft.fft(Hc, dim=-1)
        peak = torch.argmax(torch.abs(Hf), dim=-1)  # [B,K]
        n = torch.arange(N, device=H0.device).float()
        offs = torch.arange(self.M, device=H0.device).float() - self.M // 2
        freq = (peak.unsqueeze(-1) + offs) / N  # [B,K,M]
        dict_ = torch.exp(2j * torch.pi * freq.unsqueeze(-1) * n) / np.sqrt(N)  # [B,K,M,N]
        mixD = torch.einsum('bkm,bkmn->bkn', beta.to(dict_.dtype), dict_)
        Wc = mixH + mixD
        V = W_to_V(Wc, p)
        return V, H_rate, cls, clh, p


class CNN_Wres(nn.Module):
    """w_k = normalize(h_k + scale * residual_from_features)"""
    def __init__(self):
        super().__init__()
        self.trunk = Trunk(12)
        self.fc = nn.Linear(12 * N * K, K + K + K * 2 * N)

    def forward(self, H0, cl):
        cls = cl.squeeze(-1)
        Hc, H_rate, mean, std = prep_H(H0)
        x = (H0 - mean) / std
        x = rearrange(x, 'b k n a -> b a n k')
        feat = self.trunk(x, N, K)
        out = self.fc(feat)
        p = torch.sigmoid(out[:, :K])
        clh = torch.sigmoid(out[:, K:2 * K])
        res = out[:, 2 * K:].reshape(-1, K, N, 2)
        p = c3_project(p, cls)
        Rc = torch.complex(res[..., 0], res[..., 1])
        sc = 0.3 * torch.norm(Hc, dim=-1, keepdim=True)
        Wc = Hc + sc * Rc
        V = W_to_V(Wc, p)
        return V, H_rate, cls, clh, p


def run_model(model, H0, cl, train=False, opt=None, crit_mse=None):
    V, H_rate, cls, clh, p = model(H0, cl)
    Rsum, single = SMR_loss(V, H_rate, SIGMA)
    penalty = torch.relu(0.6 - single) * 10.0
    loss_pre = -torch.mean(Rsum) + torch.mean(torch.sum(penalty, dim=1))
    loss_cl = crit_mse(clh, cls)
    loss = loss_pre + 5.0 * loss_cl
    rate = torch.mean(Rsum).item()
    acc = ((clh >= 0.5).float() == cls).float().mean().item()
    return loss, rate, acc


def train_one(name, model, epochs, lr=1e-4, batch_size=100):
    print(f'\n======== {name} epochs={epochs} ========')
    model.to(device)
    print(f'params={sum(p.numel() for p in model.parameters())/1e6:.3f}M')
    tr = torch.utils.data.DataLoader(ChannelDataset(data_root, is_train=1), batch_size=batch_size, shuffle=True)
    va = torch.utils.data.DataLoader(ChannelDataset(data_root, is_train=0), batch_size=batch_size, shuffle=False)
    te = torch.utils.data.DataLoader(ChannelDataset(data_root, is_train=2), batch_size=batch_size, shuffle=False)
    opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=1e-4)
    crit_mse = nn.MSELoss()
    best = -1e9
    best_state = None
    history = []
    for ep in range(epochs):
        model.train()
        t0 = time.time()
        for data in tr:
            opt.zero_grad()
            loss, _, _ = run_model(model, data['H'].to(device), data['cl'].to(device), True, opt, crit_mse)
            loss.backward()
            opt.step()
        model.eval()
        rates, accs = [], []
        with torch.no_grad():
            for data in va:
                _, r, a = run_model(model, data['H'].to(device), data['cl'].to(device), False, None, crit_mse)
                rates.append(r); accs.append(a)
        vr, va_ = float(np.mean(rates)), float(np.mean(accs))
        history.append({'ep': ep + 1, 'val_rate': vr, 'val_acc': va_})
        if vr > best:
            best = vr
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        print(f'  ep{ep+1}/{epochs} val={vr:.4f} acc={va_:.3f} t={time.time()-t0:.0f}s')
    model.load_state_dict(best_state)
    model.eval()
    rates, accs = [], []
    with torch.no_grad():
        for data in te:
            _, r, a = run_model(model, data['H'].to(device), data['cl'].to(device), False, None, crit_mse)
            rates.append(r); accs.append(a)
    te_r, te_a = float(np.mean(rates)), float(np.mean(accs))
    print(f'  TEST {te_r:.4f} acc={te_a:.3f} gap={GPT2_REF-te_r:+.4f}')
    return {'name': name, 'test_rate': te_r, 'test_acc': te_a, 'gap': GPT2_REF - te_r,
            'best_val': best, 'history': history}


def main():
    results = []
    # 1) longer full-W
    results.append(train_one('whd_long', CNN_Whd(), epochs=40))
    # 2) medium
    results.append(train_one('wmed', CNN_Wmed(M=16), epochs=25))
    # 3) residual
    results.append(train_one('wres', CNN_Wres(), epochs=25))
    out = os.path.join(base_output, 'mini_exp_E2b.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print('\n===== SUMMARY =====')
    for r in results:
        print(f"{r['name']:10s} test={r['test_rate']:.4f} acc={r['test_acc']:.3f} gap={r['gap']:+.4f}")
    print('saved', out)


if __name__ == '__main__':
    main()
