"""
DL 间隙小实验：短训 CNN 变体，看谁能让 rate 相对 GPT2 31.92 掉下来。
每个变体 epochs 默认 15，协议与主线一致（MULoss + gamma2*MSE）。

变体:
  base     — 当前 CNN_pre (对照)
  softc3   — C3 用 sigmoid(cl_hat) 软标签（分类进入 rate 路径）
  whd      — 输出高维 W (2N per user) + p + cl，不再共用低维 (p,λ)
  tiny     — 通道数 12→4（降低容量）
  weakreg  — base 但 lr=3e-4 更容易欠拟合 rate（对照优化）
"""
import os, sys, argparse, time, datetime, glob, json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange

sys.path.insert(0, r"C:\Users\17859\Desktop\files\Grad_Project\LLM for LAE\Codes_v1")
from data import ChannelDataset
from utils import ACCLoss, RateCal, MULoss, pq2V
from models.baseline_CNN import CNN_pre

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda:0')
N, K = 256, 10
data_root = r"C:\Users\17859\Desktop\files\Grad_Project\LLM for LAE\Codes_v1\Data_user.mat"
base_output = r"C:\Users\17859\Desktop\files\Grad_Project\LLM for LAE\Codes_v1\output"
GPT2_REF = 31.9208  # 官方 GPT2 test rate


class CNN_SoftC3(nn.Module):
    """与 CNN_pre 相同，但 C3 用软分类概率。"""
    def __init__(self, N, K, gamma):
        super().__init__()
        self.inner = CNN_pre(N, K, gamma)
        self.gamma = gamma
        self.P_max = 1

    def forward(self, x, cl, index, mean, std, use_pred_cl_for_c3=True):
        # bypass inner C3: call conv trunk manually by wrapping
        # simpler: patch inner forward with use_pred_cl and soft mask
        x = (x - mean) / std
        x = F.relu(self.inner.bn1(self.inner.conv1(x)))
        x = F.relu(self.inner.bn2(self.inner.conv2(x)))
        x = x.view(-1, 12 * self.inner.N * self.inner.K)
        x = torch.sigmoid(self.inner.fc(x))
        x = rearrange(x, 'n (H a) -> n H a', a=3)
        cl = cl.squeeze(-1)
        p_hat = x[:, :index, 0]
        lamda_hat = x[:, :index, 1]
        cl_hat = x[:, :index, 2]
        # soft C3 label
        cl_c3 = cl_hat  # in [0,1], continuous
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
        p_hat_0 = mask0 * p_normalized + (~mask0) * (normalized_label_1 + normalized_label_0)
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
        lamda_hat_0 = mask1 * lamda_normalized + (~mask1) * (normalized_P_label_1 + normalized_P_label_0)
        return p_hat_0, lamda_hat_0, cl_hat


class CNN_Whd(nn.Module):
    """CNN 输出高维 W[K,2N] + p[K] + cl[K]，不再学 λ。"""
    def __init__(self, N, K, gamma):
        super().__init__()
        self.N, self.K, self.gamma = N, K, gamma
        self.P_max = 1
        self.conv1 = nn.Conv2d(2, 12, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(12, momentum=0.99, eps=0.001)
        self.conv2 = nn.Conv2d(12, 12, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(12, momentum=0.99, eps=0.001)
        self.fc = nn.Linear(12 * N * K, K + K + K * 2 * N)

    def forward(self, x, cl, index, mean, std, use_pred_cl_for_c3=False):
        x = (x - mean) / std
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.view(-1, 12 * self.N * self.K)
        out = self.fc(x)
        p_hat = torch.sigmoid(out[:, :index])
        rest = out[:, index:]
        cl_hat = torch.sigmoid(rest[:, :index])
        W = rest[:, index:].reshape(-1, index, self.N, 2)  # raw W re/im
        cl = cl.squeeze(-1)
        cl_c3 = (cl_hat >= 0.5).float() if use_pred_cl_for_c3 else cl
        # C3 on p (same as official)
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
        p_hat_0 = mask0 * p_normalized + (~mask0) * (normalized_label_1 + normalized_label_0)
        return W, p_hat_0, cl_hat

    def to_V(self, W, p_hat, H):
        """W [B,K,N,2] raw -> V [B,N,1,K,2] with ||v_k||^2 = p_k, total power 1."""
        Wc = torch.complex(W[..., 0], W[..., 1])  # [B,K,N]
        Wc = Wc / (torch.norm(Wc, dim=-1, keepdim=True) + 1e-24)
        Vc = (Wc * p_hat.unsqueeze(-1)).permute(0, 2, 1).unsqueeze(2)  # [B,N,1,K]
        V = torch.stack((Vc.real, Vc.imag), dim=-1)
        cur = torch.sum(torch.abs(Vc) ** 2, dim=(1, 2, 3))
        V = V * torch.sqrt(1.0 / (cur + 1e-24)).reshape(-1, 1, 1, 1, 1)
        return V


class CNN_Tiny(CNN_pre):
    def __init__(self, N, K, gamma):
        nn.Module.__init__(self)
        self.N, self.K, self.gamma = N, K, gamma
        self.P_max = 1
        self.conv1 = nn.Conv2d(2, 4, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(4, momentum=0.99, eps=0.001)
        self.conv2 = nn.Conv2d(4, 4, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(4, momentum=0.99, eps=0.001)
        self.fc = nn.Linear(4 * N * K, 3 * K)

    def forward(self, x, cl, index, mean, std, use_pred_cl_for_c3=False):
        x = (x - mean) / std
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.view(-1, 4 * self.N * self.K)
        x = torch.sigmoid(self.fc(x))
        x = rearrange(x, 'n (H a) -> n H a', a=3)
        cl = cl.squeeze(-1)
        p_hat = x[:, :index, 0]
        lamda_hat = x[:, :index, 1]
        cl_hat = x[:, :index, 2]
        cl_c3 = cl
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
        p_hat_0 = mask0 * p_normalized + (~mask0) * (normalized_label_1 + normalized_label_0)
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
        lamda_hat_0 = mask1 * lamda_normalized + (~mask1) * (normalized_P_label_1 + normalized_P_label_0)
        return p_hat_0, lamda_hat_0, cl_hat


def build_model(variant):
    if variant == 'base':
        return CNN_pre(N, K, 0.4), 'pl'
    if variant == 'softc3':
        return CNN_SoftC3(N, K, 0.4), 'pl'
    if variant == 'whd':
        return CNN_Whd(N, K, 0.4), 'w'
    if variant == 'tiny':
        return CNN_Tiny(N, K, 0.4), 'pl'
    if variant == 'weakreg':
        return CNN_pre(N, K, 0.4), 'pl'
    raise ValueError(variant)


def forward_pl(model, H0, cl, K):
    H_re = rearrange(H0, 'n k W H -> n H W k', H=2)
    H0z = torch.zeros(H_re.shape, device=H0.device)
    Hs = H_re[:, :, :, :K]
    cls = cl[:, :K]
    mean, std = torch.mean(Hs), torch.std(Hs)
    H0z[:, :, :, :K] = Hs
    p, lam, clh = model(H0z, cls, K, mean, std)
    H_rate = rearrange(H0z, 'n H W k -> n k (W H)')[:, :K, :]
    return p, lam, clh, H_rate, cls


def train_variant(variant, epochs=15, lr=1e-4, batch_size=100):
    print(f'\n======== VARIANT {variant} epochs={epochs} lr={lr} ========')
    model, kind = build_model(variant)
    model.to(device)
    nparam = sum(p.numel() for p in model.parameters())
    print(f'params={nparam/1e6:.3f}M')

    train_set = ChannelDataset(data_root, is_train=1)
    val_set = ChannelDataset(data_root, is_train=0)
    test_set = ChannelDataset(data_root, is_train=2)
    tr = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    va = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)
    te = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=1e-4)
    crit_mu = MULoss(rmin=0.6).to(device)
    crit_rate = RateCal().to(device)
    crit_mse = nn.MSELoss().to(device)
    crit_acc = ACCLoss().to(device)

    best = -1e9
    for ep in range(epochs):
        model.train()
        t0 = time.time()
        for data in tr:
            opt.zero_grad()
            H0 = data['H'].to(device)
            cl = data['cl'].to(device)
            if kind == 'pl':
                p, lam, clh, H_rate, cls = forward_pl(model, H0, cl, K)
                loss_pre = crit_mu(p, lam, H_rate)
                loss_cl = crit_mse(cls, torch.unsqueeze(clh, 2))
                loss = loss_pre + 5.0 * loss_cl
            else:
                H_re = rearrange(H0, 'n k W H -> n H W k', H=2)
                H0z = torch.zeros(H_re.shape, device=device)
                Hs = H_re[:, :, :, :K]; cls = cl[:, :K]
                mean, std = torch.mean(Hs), torch.std(Hs)
                H0z[:, :, :, :K] = Hs
                W, p, clh = model(H0z, cls, K, mean, std)
                H_rate = rearrange(H0z, 'n H W k -> n k (W H)')[:, :K, :]
                V = model.to_V(W, p, H_rate)
                # reuse MULoss via fake lam: build equivalent? use RateCal path
                # compute rate from V directly
                from utils import SMR_loss
                sigma = 1e-2
                Rsum, single = SMR_loss(V, H_rate, sigma)
                penalty = torch.relu(0.6 - single) * 10.0
                loss_pre = -torch.mean(Rsum) + torch.mean(torch.sum(penalty, dim=1))
                loss_cl = crit_mse(cls, torch.unsqueeze(clh, 2))
                loss = loss_pre + 5.0 * loss_cl
            loss.backward()
            opt.step()
        # val
        model.eval()
        rates, accs = [], []
        with torch.no_grad():
            for data in va:
                H0 = data['H'].to(device); cl = data['cl'].to(device)
                if kind == 'pl':
                    p, lam, clh, H_rate, cls = forward_pl(model, H0, cl, K)
                    rates.append(crit_rate(p, lam, H_rate).item())
                    accs.append(crit_acc(cls, torch.unsqueeze(clh, 2)).item())
                else:
                    H_re = rearrange(H0, 'n k W H -> n H W k', H=2)
                    H0z = torch.zeros(H_re.shape, device=device)
                    Hs = H_re[:, :, :, :K]; cls = cl[:, :K]
                    mean, std = torch.mean(Hs), torch.std(Hs)
                    H0z[:, :, :, :K] = Hs
                    W, p, clh = model(H0z, cls, K, mean, std)
                    H_rate = rearrange(H0z, 'n H W k -> n k (W H)')[:, :K, :]
                    V = model.to_V(W, p, H_rate)
                    from utils import SMR_loss
                    Rsum, _ = SMR_loss(V, H_rate, 1e-2)
                    rates.append(torch.mean(Rsum).item())
                    accs.append(crit_acc(cls, torch.unsqueeze(clh, 2)).item())
        vr, va_ = np.mean(rates), np.mean(accs)
        if vr > best:
            best = vr
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        print(f'  ep{ep+1}/{epochs} val_rate={vr:.4f} acc={va_:.4f} time={time.time()-t0:.1f}s')

    model.load_state_dict(best_state)
    model.eval()
    rates, accs = [], []
    with torch.no_grad():
        for data in te:
            H0 = data['H'].to(device); cl = data['cl'].to(device)
            if kind == 'pl':
                p, lam, clh, H_rate, cls = forward_pl(model, H0, cl, K)
                rates.append(crit_rate(p, lam, H_rate).item())
                accs.append(crit_acc(cls, torch.unsqueeze(clh, 2)).item())
            else:
                H_re = rearrange(H0, 'n k W H -> n H W k', H=2)
                H0z = torch.zeros(H_re.shape, device=device)
                Hs = H_re[:, :, :, :K]; cls = cl[:, :K]
                mean, std = torch.mean(Hs), torch.std(Hs)
                H0z[:, :, :, :K] = Hs
                W, p, clh = model(H0z, cls, K, mean, std)
                H_rate = rearrange(H0z, 'n H W k -> n k (W H)')[:, :K, :]
                V = model.to_V(W, p, H_rate)
                from utils import SMR_loss
                Rsum, _ = SMR_loss(V, H_rate, 1e-2)
                rates.append(torch.mean(Rsum).item())
                accs.append(crit_acc(cls, torch.unsqueeze(clh, 2)).item())
    te_r, te_a = float(np.mean(rates)), float(np.mean(accs))
    gap = GPT2_REF - te_r
    print(f'  TEST rate={te_r:.4f} acc={te_a:.4f}  GPT2_ref-GAP={gap:+.4f}')
    return {'variant': variant, 'epochs': epochs, 'lr': lr, 'params': nparam,
            'val_rate': float(best), 'test_rate': te_r, 'test_acc': te_a, 'gap_vs_gpt2': gap}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--variants', type=str, default='base,softc3,whd,tiny')
    args = ap.parse_args()
    results = []
    for v in args.variants.split(','):
        v = v.strip()
        lr = 3e-4 if v == 'weakreg' else 1e-4
        results.append(train_variant(v, epochs=args.epochs, lr=lr))
        out = os.path.join(base_output, 'mini_exp_dl_gap.json')
        with open(out, 'w') as f:
            json.dump(results, f, indent=2)
        print(f'saved {out}')
    print('\n===== SUMMARY =====')
    for r in results:
        print(f"{r['variant']:8s} test={r['test_rate']:.4f} acc={r['test_acc']:.3f} gap={r['gap_vs_gpt2']:+.4f}")


if __name__ == '__main__':
    main()
