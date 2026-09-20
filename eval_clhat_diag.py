"""
eval_clhat_diag.py — Diagnostic only.

Compare spectrum efficiency when C3 power projection uses:
  (A) ground-truth cl  (current official eval path)
  (B) predicted cl_hat >= 0.5  (diagnostic)

Does NOT change training or official result JSONs.
"""
import os, sys, json, glob
import torch
import numpy as np
from einops import rearrange
from models.gpt2_model_all import Gpt2Model
from models.baseline_CNN import CNN_pre
from data import ChannelDataset
from utils import ACCLoss, RateCal, pq2V

sys.stdout.reconfigure(encoding='utf-8')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda:0')
N, K = 256, 10
batch_size = 100
gamma = 0.4

gpt2_model_path = r"C:\Users\17859\.cache\huggingface\hub\models--openai-community--gpt2\snapshots\607a30d783dfa663caf39e06633721c8d4cfcd7e"
data_root = r"C:\Users\17859\Desktop\files\Grad_Project\LLM for LAE\Codes_v1\Data_user.mat"
base_output = r"C:\Users\17859\Desktop\files\Grad_Project\LLM for LAE\Codes_v1\output"


def load_test_loader():
    test_set = ChannelDataset(data_root, is_train=2, train_per=0.8, valid_per=0.1)
    return torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)


def find_paper_default(prefix, ext):
    runs = sorted(glob.glob(os.path.join(base_output, f"{prefix}_*")))
    valid = []
    for r in runs:
        name = os.path.basename(r).lower()
        if not glob.glob(os.path.join(r, f'*.{ext}')):
            continue
        if 'gamma' in name or 'rmin' in name:
            continue
        valid.append(r)
    if not valid:
        raise FileNotFoundError(f"No {prefix} run")
    run_dir = valid[-1]
    ckpts = sorted(glob.glob(os.path.join(run_dir, f'*.{ext}')), key=os.path.getmtime)
    return run_dir, ckpts[-1]


def eval_gpt2(model, loader, use_pred_cl):
    criterion_rate = RateCal().to(device)
    criterion_acc = ACCLoss().to(device)
    rates, accs, alpha_Ns = [], [], []
    with torch.no_grad():
        for data in loader:
            H = data['H'].to(device, non_blocking=True)
            H = rearrange(H, 'n W H a -> n W (H a)')
            cl = data['cl'].to(device, non_blocking=True)
            p_hat, lamda_hat, cl_hat = model(H, cl, use_pred_cl_for_c3=use_pred_cl)
            rates.append(criterion_rate(p_hat, lamda_hat, H).item())
            accs.append(criterion_acc(cl, torch.unsqueeze(cl_hat, dim=2)).item())
            V = pq2V(p_hat, lamda_hat, H, 0.01, N)
            Vc = torch.view_as_complex(V.contiguous())
            power = torch.sum(torch.abs(Vc) ** 2, dim=(1, 2))
            cl_s = cl.squeeze(-1)
            alpha_Ns.append((torch.sum(power * cl_s, dim=1) / (torch.sum(power, dim=1) + 1e-8)).mean().item())
    return float(np.mean(rates)), float(np.mean(accs)), float(np.mean(alpha_Ns))


def eval_cnn(model, loader, use_pred_cl):
    criterion_rate = RateCal().to(device)
    criterion_acc = ACCLoss().to(device)
    rates, accs, alpha_Ns = [], [], []
    with torch.no_grad():
        for data in loader:
            H = data['H'].to(device, non_blocking=True)
            cl = data['cl'].to(device, non_blocking=True)
            H_re = rearrange(H, 'n k W H -> n H W k', H=2)
            H0 = torch.zeros(H_re.shape, device=device)
            H_s = H_re[:, :, :, :K]
            cl_s = cl[:, :K, :]
            mean = torch.mean(H_s)
            std = torch.std(H_s)
            H0[:, :, :, :K] = H_s
            p_hat, lamda_hat, cl_hat = model(H0, cl_s, K, mean, std, use_pred_cl_for_c3=use_pred_cl)
            H_rate = rearrange(H0, 'n H W k -> n k (W H)')[:, :K, :]
            rates.append(criterion_rate(p_hat, lamda_hat, H_rate).item())
            accs.append(criterion_acc(cl_s, torch.unsqueeze(cl_hat, dim=2)).item())
            V = pq2V(p_hat, lamda_hat, H_rate, 0.01, N)
            Vc = torch.view_as_complex(V.contiguous())
            power = torch.sum(torch.abs(Vc) ** 2, dim=(1, 2))
            cls = cl_s.squeeze(-1)
            alpha_Ns.append((torch.sum(power * cls, dim=1) / (torch.sum(power, dim=1) + 1e-8)).mean().item())
    return float(np.mean(rates)), float(np.mean(accs)), float(np.mean(alpha_Ns))


def main():
    loader = load_test_loader()
    print(f"Test samples: {len(loader.dataset)}, gamma={gamma}")

    results = {'gamma': gamma, 'gpt2': {}, 'cnn': {}}

    gpt2_run, gpt2_ckpt = find_paper_default('GPT2', 'bin')
    print(f"\nGPT2 run: {gpt2_run}")
    print(f"GPT2 ckpt: {os.path.basename(gpt2_ckpt)}")
    gpt2 = Gpt2Model(model_path=gpt2_model_path, Nt=N, K=K, gamma=gamma)
    gpt2.to(device)
    gpt2.load_state_dict(torch.load(gpt2_ckpt, map_location=device, weights_only=True))
    gpt2.eval()

    for use_pred, tag in [(False, 'gt_cl'), (True, 'pred_cl_hat')]:
        r, a, an = eval_gpt2(gpt2, loader, use_pred)
        results['gpt2'][tag] = {'rate': r, 'acc': a, 'alpha_N_true_group': an}
        print(f"GPT2 C3={tag}: rate={r:.4f} acc={a:.4f} alpha_N(by GT label)={an:.4f}")

    cnn_run, cnn_ckpt = find_paper_default('CNN', 'pth')
    print(f"\nCNN run: {cnn_run}")
    print(f"CNN ckpt: {os.path.basename(cnn_ckpt)}")
    cnn = CNN_pre(N, K, gamma)
    cnn.to(device)
    cnn.load_state_dict(torch.load(cnn_ckpt, map_location=device, weights_only=True))
    cnn.eval()

    for use_pred, tag in [(False, 'gt_cl'), (True, 'pred_cl_hat')]:
        r, a, an = eval_cnn(cnn, loader, use_pred)
        results['cnn'][tag] = {'rate': r, 'acc': a, 'alpha_N_true_group': an}
        print(f"CNN  C3={tag}: rate={r:.4f} acc={a:.4f} alpha_N(by GT label)={an:.4f}")

    gap_gt = results['gpt2']['gt_cl']['rate'] - results['cnn']['gt_cl']['rate']
    gap_pred = results['gpt2']['pred_cl_hat']['rate'] - results['cnn']['pred_cl_hat']['rate']
    drop_g = results['gpt2']['gt_cl']['rate'] - results['gpt2']['pred_cl_hat']['rate']
    drop_c = results['cnn']['gt_cl']['rate'] - results['cnn']['pred_cl_hat']['rate']
    results['summary'] = {
        'gap_gt_cl': gap_gt,
        'gap_pred_cl': gap_pred,
        'gpt2_drop_when_pred_c3': drop_g,
        'cnn_drop_when_pred_c3': drop_c,
        'paper_gap_ref': 31.10 - 30.59,
    }
    print('\n===== SUMMARY =====')
    print(f"Gap (C3=gt_cl):        {gap_gt:+.4f}  (paper ref ~ +0.51)")
    print(f"Gap (C3=pred_cl_hat):  {gap_pred:+.4f}")
    print(f"GPT2 drop if C3=pred:  {drop_g:+.4f}")
    print(f"CNN  drop if C3=pred:  {drop_c:+.4f}")

    out = os.path.join(base_output, 'eval_clhat_diag_results.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == '__main__':
    main()
