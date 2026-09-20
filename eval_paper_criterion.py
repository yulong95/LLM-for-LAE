"""
eval_paper_criterion.py — Diagnostic: select/compare checkpoints by paper total loss.

Paper: Loss = gamma2 * Loss_cl + Loss_pre
  Loss_pre = -sum(R) + gamma1 * ||max(Rmin-R,0)||_1   (MULoss)
  Loss_cl  = ||X_cl - X_cl_hat||^2                    (MSE)

Current official training selects by max val RateCal only.
This script evaluates existing paper-default checkpoints under BOTH criteria.
Does not retrain.
"""
import os, sys, json, glob
import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
from models.gpt2_model_all import Gpt2Model
from models.baseline_CNN import CNN_pre
from data import ChannelDataset
from utils import ACCLoss, RateCal, MULoss

sys.stdout.reconfigure(encoding='utf-8')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda:0')
N, K = 256, 10
batch_size = 100
gamma = 0.4
gamma2 = 5.0
rmin = 0.6
gamma1 = 10.0

gpt2_model_path = r"C:\Users\17859\.cache\huggingface\hub\models--openai-community--gpt2\snapshots\607a30d783dfa663caf39e06633721c8d4cfcd7e"
data_root = r"C:\Users\17859\Desktop\files\Grad_Project\LLM for LAE\Codes_v1\Data_user.mat"
base_output = r"C:\Users\17859\Desktop\files\Grad_Project\LLM for LAE\Codes_v1\output"


def load_loader(is_train=2):
    ds = ChannelDataset(data_root, is_train=is_train, train_per=0.8, valid_per=0.1)
    return torch.utils.data.DataLoader(dataset=ds, batch_size=batch_size, shuffle=False)


def find_paper_default_runs(prefix):
    runs = sorted(glob.glob(os.path.join(base_output, f"{prefix}_*")))
    valid = []
    for r in runs:
        name = os.path.basename(r).lower()
        if 'gamma' in name or 'rmin' in name:
            continue
        valid.append(r)
    return valid


def collect_ckpts(prefix, ext):
    """Return list of (run_dir, ckpt_path, tag)."""
    out = []
    for run in find_paper_default_runs(prefix):
        files = sorted(glob.glob(os.path.join(run, f'*.{ext}')), key=os.path.getmtime)
        for f in files:
            tag = f"{os.path.basename(run)}/{os.path.basename(f)}"
            out.append((run, f, tag))
    return out


def eval_gpt2_ckpt(ckpt, loader):
    model = Gpt2Model(model_path=gpt2_model_path, Nt=N, K=K, gamma=gamma)
    model.to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()
    criterion_rate = RateCal().to(device)
    criterion_acc = ACCLoss().to(device)
    criterion_mu = MULoss(rmin=rmin, gamma1=gamma1).to(device)
    mse = nn.MSELoss()
    rates, accs, mus, cls = [], [], [], []
    with torch.no_grad():
        for data in loader:
            H = data['H'].to(device, non_blocking=True)
            H = rearrange(H, 'n W H a -> n W (H a)')
            cl = data['cl'].to(device, non_blocking=True)
            p_hat, lamda_hat, cl_hat = model(H, cl)
            rates.append(criterion_rate(p_hat, lamda_hat, H).item())
            accs.append(criterion_acc(cl, torch.unsqueeze(cl_hat, dim=2)).item())
            mus.append(criterion_mu(p_hat, lamda_hat, H).item())
            cls.append(mse(cl, torch.unsqueeze(cl_hat, dim=2)).item())
    rate = float(np.mean(rates))
    acc = float(np.mean(accs))
    loss_pre = float(np.mean(mus))  # already -sumR + penalty
    loss_cl = float(np.mean(cls))
    total = loss_pre + gamma2 * loss_cl
    return {'rate': rate, 'acc': acc, 'loss_pre': loss_pre, 'loss_cl': loss_cl, 'loss_total': total}


def eval_cnn_ckpt(ckpt, loader):
    model = CNN_pre(N, K, gamma)
    model.to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()
    criterion_rate = RateCal().to(device)
    criterion_acc = ACCLoss().to(device)
    criterion_mu = MULoss(rmin=rmin, gamma1=gamma1).to(device)
    mse = nn.MSELoss()
    rates, accs, mus, cls = [], [], [], []
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
            p_hat, lamda_hat, cl_hat = model(H0, cl_s, K, mean, std)
            H_rate = rearrange(H0, 'n H W k -> n k (W H)')[:, :K, :]
            rates.append(criterion_rate(p_hat, lamda_hat, H_rate).item())
            accs.append(criterion_acc(cl_s, torch.unsqueeze(cl_hat, dim=2)).item())
            mus.append(criterion_mu(p_hat, lamda_hat, H_rate).item())
            cls.append(mse(cl_s, torch.unsqueeze(cl_hat, dim=2)).item())
    rate = float(np.mean(rates))
    acc = float(np.mean(accs))
    loss_pre = float(np.mean(mus))
    loss_cl = float(np.mean(cls))
    total = loss_pre + gamma2 * loss_cl
    return {'rate': rate, 'acc': acc, 'loss_pre': loss_pre, 'loss_cl': loss_cl, 'loss_total': total}


def main():
    test_loader = load_loader(2)
    val_loader = load_loader(0)
    print(f"Test={len(test_loader.dataset)}, Val={len(val_loader.dataset)}")
    print(f"gamma={gamma}, gamma2={gamma2}, rmin={rmin}, gamma1={gamma1}")
    print("Selection metric (paper): minimize loss_total = loss_pre + gamma2*loss_cl")
    print("Official metric (current code): maximize rate (RateCal)\n")

    results = {'params': {'gamma': gamma, 'gamma2': gamma2, 'rmin': rmin}, 'gpt2': [], 'cnn': []}

    gpt2_ckpts = collect_ckpts('GPT2', 'bin')
    print(f"GPT2 checkpoints found: {len(gpt2_ckpts)}")
    for run, ckpt, tag in gpt2_ckpts:
        print(f"  eval GPT2 {tag} ...")
        m_val = eval_gpt2_ckpt(ckpt, val_loader)
        m_test = eval_gpt2_ckpt(ckpt, test_loader)
        rec = {'tag': tag, 'val': m_val, 'test': m_test}
        results['gpt2'].append(rec)
        print(f"    val : rate={m_val['rate']:.4f} acc={m_val['acc']:.4f} "
              f"Lpre={m_val['loss_pre']:.4f} Lcl={m_val['loss_cl']:.6f} Ltot={m_val['loss_total']:.4f}")
        print(f"    test: rate={m_test['rate']:.4f} acc={m_test['acc']:.4f} Ltot={m_test['loss_total']:.4f}")

    cnn_ckpts = collect_ckpts('CNN', 'pth')
    print(f"\nCNN checkpoints found: {len(cnn_ckpts)}")
    for run, ckpt, tag in cnn_ckpts:
        print(f"  eval CNN {tag} ...")
        m_val = eval_cnn_ckpt(ckpt, val_loader)
        m_test = eval_cnn_ckpt(ckpt, test_loader)
        rec = {'tag': tag, 'val': m_val, 'test': m_test}
        results['cnn'].append(rec)
        print(f"    val : rate={m_val['rate']:.4f} acc={m_val['acc']:.4f} "
              f"Lpre={m_val['loss_pre']:.4f} Lcl={m_val['loss_cl']:.6f} Ltot={m_val['loss_total']:.4f}")
        print(f"    test: rate={m_test['rate']:.4f} acc={m_test['acc']:.4f} Ltot={m_test['loss_total']:.4f}")

    def pick(items, key, mode):
        if not items:
            return None
        if mode == 'min':
            return min(items, key=lambda r: r['val'][key])
        return max(items, key=lambda r: r['val'][key])

    print('\n===== SELECTION COMPARISON =====')
    summary = {}
    for model_name, items in [('GPT2', results['gpt2']), ('CNN', results['cnn'])]:
        if not items:
            print(f"{model_name}: no ckpts")
            continue
        by_rate = pick(items, 'rate', 'max')
        by_total = pick(items, 'loss_total', 'min')
        summary[model_name] = {
            'best_by_rate': {'tag': by_rate['tag'], 'test_rate': by_rate['test']['rate'],
                             'test_acc': by_rate['test']['acc'], 'val_Ltot': by_rate['val']['loss_total']},
            'best_by_paper_loss': {'tag': by_total['tag'], 'test_rate': by_total['test']['rate'],
                                   'test_acc': by_total['test']['acc'], 'val_Ltot': by_total['val']['loss_total']},
        }
        print(f"\n{model_name}:")
        print(f"  official (max val rate):  {by_rate['tag']}")
        print(f"    test rate={by_rate['test']['rate']:.4f} acc={by_rate['test']['acc']:.4f} val_Ltot={by_rate['val']['loss_total']:.4f}")
        print(f"  paper   (min val Ltot):   {by_total['tag']}")
        print(f"    test rate={by_total['test']['rate']:.4f} acc={by_total['test']['acc']:.4f} val_Ltot={by_total['val']['loss_total']:.4f}")
        print(f"  same ckpt? {by_rate['tag'] == by_total['tag']}")

    if 'GPT2' in summary and 'CNN' in summary:
        gap_official = summary['GPT2']['best_by_rate']['test_rate'] - summary['CNN']['best_by_rate']['test_rate']
        gap_paper = summary['GPT2']['best_by_paper_loss']['test_rate'] - summary['CNN']['best_by_paper_loss']['test_rate']
        print(f"\nRate gap official selection: {gap_official:+.4f}")
        print(f"Rate gap paper-loss selection: {gap_paper:+.4f}")
        print(f"Paper ref gap: +0.51")
        summary['gaps'] = {'official': gap_official, 'paper_loss': gap_paper}

    results['summary'] = summary
    out = os.path.join(base_output, 'eval_paper_criterion_results.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == '__main__':
    main()
