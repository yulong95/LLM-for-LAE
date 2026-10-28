"""
eval_amix.py — A-mix 模型统一评估
输出: eval_amix_{gpt2,tf,cnn}_results.json
含 K-sweep / SNR / P-sweep / alpha_c sweep
"""
import os, sys, argparse, glob, json
import torch
import numpy as np
from einops import rearrange

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import ChannelDataset
from utils import ACCLoss, SMR_loss
from models.amix_models import TF_Amix, CNN_Amix, GPT2_Amix

sys.stdout.reconfigure(encoding='utf-8')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda:0')
N, K = 256, 10
batch_size = 100
SIGMA = 1e-2
data_root = r"C:\Users\17859\Desktop\files\Grad_Project\LLM for LAE\Codes_v1\Data_user.mat"
base_output = r"C:\Users\17859\Desktop\files\Grad_Project\LLM for LAE\Codes_v1\output"
gpt2_model_path = r"C:\Users\17859\.cache\huggingface\hub\models--openai-community--gpt2\snapshots\607a30d783dfa663caf39e06633721c8d4cfcd7e"

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True, choices=['gpt2', 'tf', 'cnn'])
parser.add_argument('--run', type=str, default=None, help='checkpoint dir; default latest')
args = parser.parse_args()


def find_run():
    if args.run:
        return args.run
    tag = {'gpt2': 'GPT2Amix', 'tf': 'TFAmix', 'cnn': 'CNNAmix'}[args.model]
    runs = sorted(glob.glob(os.path.join(base_output, f'{tag}_*')))
    runs = [r for r in runs if glob.glob(os.path.join(r, 'best.pth'))]
    if not runs:
        raise FileNotFoundError(f'no {tag} run')
    return runs[-1]


def build_model(gamma=0.4):
    if args.model == 'tf':
        model = TF_Amix(gamma=gamma)
    elif args.model == 'cnn':
        model = CNN_Amix(gamma=gamma)
    else:
        model = GPT2_Amix(gpt2_model_path, gamma=gamma)
    return model.to(device)


def load_test():
    ds = ChannelDataset(data_root, is_train=2, train_per=0.8, valid_per=0.1)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)


def add_csi_awgn(H_flat, snr_db):
    H_4d = H_flat.reshape(*H_flat.shape[:-1], H_flat.shape[-1] // 2, 2)
    H_complex = torch.complex(H_4d[..., 0], H_4d[..., 1])
    p_sig = torch.mean(torch.abs(H_complex) ** 2)
    noise_cplx_power = p_sig / (10 ** (snr_db / 10.0))
    n_complex = (torch.randn_like(H_complex) + 1j * torch.randn_like(H_complex)) * torch.sqrt(noise_cplx_power / 2.0)
    noise = torch.stack([n_complex.real, n_complex.imag], dim=-1).reshape_as(H_flat)
    return H_flat + noise


def eval_k(model, loader, K_eval):
    """Pad to K=10 for fixed A-mix head; score only first K_eval users."""
    criterion_acc = ACCLoss().to(device)
    rates, accs = [], []
    with torch.no_grad():
        for data in loader:
            H0 = data['H'].to(device)
            cl = data['cl'].to(device)
            B = H0.shape[0]
            H0p = torch.zeros(B, K, N, 2, device=device)
            clp = torch.zeros(B, K, 1, device=device)
            H0p[:, :K_eval] = H0[:, :K_eval]
            clp[:, :K_eval] = cl[:, :K_eval]
            V, H_rate, clh, p, A = model(H0p, clp)
            Rsum, _ = SMR_loss(V[:, :, :, :K_eval, :], H_rate[:, :K_eval, :], SIGMA)
            rates.append(torch.mean(Rsum).item())
            accs.append(criterion_acc(clp[:, :K_eval], torch.unsqueeze(clh[:, :K_eval], 2)).item())
    return float(np.mean(rates)), float(np.mean(accs))


def hrate_from(H0):
    return H0.reshape(H0.shape[0], H0.shape[1], -1)


def eval_snr(model, loader, snr_db):
    criterion_acc = ACCLoss().to(device)
    rates, accs = [], []
    with torch.no_grad():
        for data in loader:
            H0 = data['H'].to(device)
            cl = data['cl'].to(device)
            H_rate = hrate_from(H0)
            H_noisy = add_csi_awgn(H_rate, snr_db)
            H0n = H_noisy.reshape(H0.shape[0], H0.shape[1], N, 2)
            V, H_rate_n, clh, p, A = model(H0n, cl)
            Rsum, _ = SMR_loss(V, H_rate_n, SIGMA)
            rates.append(torch.mean(Rsum).item())
            accs.append(criterion_acc(cl, torch.unsqueeze(clh, 2)).item())
    return float(np.mean(rates)), float(np.mean(accs))


def rate_at_power(model, loader, P_total):
    rates = []
    with torch.no_grad():
        for data in loader:
            H0 = data['H'].to(device)
            cl = data['cl'].to(device)
            V, H_rate, clh, p, A = model(H0, cl)
            # scale V power to P_total
            Vc = torch.view_as_complex(V.contiguous())
            cur = torch.sum(torch.abs(Vc) ** 2, dim=(1, 2, 3))
            scale = torch.sqrt(torch.tensor(P_total, dtype=torch.float32, device=V.device) / (cur + 1e-24))
            V = V * scale.reshape(-1, 1, 1, 1, 1)
            Rsum, _ = SMR_loss(V, H_rate, SIGMA)
            rates.append(torch.mean(Rsum).item())
    return float(np.mean(rates))


def eval_gamma(model, loader, gamma):
    model.gamma = gamma
    rates = []
    with torch.no_grad():
        for data in loader:
            H0 = data['H'].to(device)
            cl = data['cl'].to(device)
            V, H_rate, clh, p, A = model(H0, cl)
            Rsum, _ = SMR_loss(V, H_rate, SIGMA)
            rates.append(torch.mean(Rsum).item())
    return float(np.mean(rates))


def main():
    run = find_run()
    print(f'Run: {run}')
    model = build_model(gamma=0.4)
    ck = os.path.join(run, 'best.pth')
    model.load_state_dict(torch.load(ck, map_location=device, weights_only=True))
    model.eval()
    loader = load_test()
    print(f'Test: {len(loader.dataset)}')

    results = {'run': run, 'model': args.model}

    print('\n===== K sweep =====')
    k_res = {}
    for k in range(5, 11):
        r, a = eval_k(model, loader, k)
        k_res[str(k)] = {'rate': r, 'acc': a}
        print(f'  K={k}: rate={r:.4f} acc={a:.4f}')
    results['k_sweep'] = k_res

    print('\n===== SNR =====')
    snr_res = {}
    for s in [0, 5, 10, 15, 20]:
        r, a = eval_snr(model, loader, s)
        snr_res[str(s)] = {'rate': r, 'acc': a}
        print(f'  SNR={s}: rate={r:.4f} acc={a:.4f}')
    results['snr_sweep'] = snr_res

    print('\n===== P sweep =====')
    p_res = {}
    for P in [-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10]:
        r = rate_at_power(model, loader, 10 ** (P / 10))
        p_res[str(P)] = r
        print(f'  P={P}: {r:.4f}')
    results['fig9'] = p_res

    print('\n===== alpha_c sweep =====')
    g_res = {}
    for g in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        r = eval_gamma(model, loader, g)
        g_res[str(g)] = r
        print(f'  gamma={g}: {r:.4f}')
    results['fig7'] = g_res

    out = os.path.join(base_output, f'eval_amix_{args.model}_results.json')
    json.dump(results, open(out, 'w'), indent=2)
    print(f'\nSaved {out}')


if __name__ == '__main__':
    main()
