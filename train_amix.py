"""
train_amix.py — 统一 A-mix 训练（GPT2 / Transformer / CNN）
用法:
  python train_amix.py --model tf
  python train_amix.py --model cnn
  python train_amix.py --model gpt2
  python train_amix.py --model tf --epochs 100
"""
import os, sys, argparse, time, datetime, json
import torch
import torch.nn as nn
import numpy as np
from einops import rearrange

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import ChannelDataset
from utils import ACCLoss, SMR_loss

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda:0')
N, K = 256, 10
SIGMA = 1e-2
data_root = r"C:\Users\17859\Desktop\files\Grad_Project\LLM for LAE\Codes_v1\Data_user.mat"
base_output = r"C:\Users\17859\Desktop\files\Grad_Project\LLM for LAE\Codes_v1\output"
gpt2_model_path = r"C:\Users\17859\.cache\huggingface\hub\models--openai-community--gpt2\snapshots\607a30d783dfa663caf39e06633721c8d4cfcd7e"

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True, choices=['gpt2', 'tf', 'cnn'])
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--gamma2', type=float, default=5.0)
parser.add_argument('--rmin', type=float, default=0.6)
parser.add_argument('--gamma', type=float, default=0.4)
args = parser.parse_args()


def build_model():
    if args.model == 'tf':
        from models.amix_models import TF_Amix
        return TF_Amix(gamma=args.gamma)
    if args.model == 'cnn':
        from models.amix_models import CNN_Amix
        return CNN_Amix(gamma=args.gamma)
    from models.amix_models import GPT2_Amix
    return GPT2_Amix(gpt2_model_path, gamma=args.gamma)


def amix_rate_loss(V, H_rate, rmin, gamma1=10.0):
    Rsum, single = SMR_loss(V, H_rate, SIGMA)
    penalty = torch.relu(rmin - single) * gamma1
    loss = -torch.mean(Rsum) + torch.mean(torch.sum(penalty, dim=1))
    return loss, torch.mean(Rsum).item(), torch.mean(((single >= rmin).float())).item()


def evaluate(model, loader, crit_acc):
    model.eval()
    rates, accs = [], []
    with torch.no_grad():
        for data in loader:
            H0 = data['H'].to(device)
            cl = data['cl'].to(device)
            V, H_rate, clh, p, A = model(H0, cl)
            _, r, _ = amix_rate_loss(V, H_rate, 0.0, 0.0)
            rates.append(r)
            accs.append(crit_acc(cl, torch.unsqueeze(clh, 2)).item())
    return float(np.mean(rates)), float(np.mean(accs))


def main():
    tag = {'gpt2': 'GPT2Amix', 'tf': 'TFAmix', 'cnn': 'CNNAmix'}[args.model]
    output_dir = os.path.join(base_output, f"{tag}_{datetime.datetime.now().strftime('%m.%d_%H-%M-%S')}")
    os.makedirs(output_dir, exist_ok=True)
    print(f'Train {tag}  epochs={args.epochs}  out={output_dir}')

    model = build_model().to(device)
    nparam = sum(p.numel() for p in model.parameters())
    print(f'params={nparam/1e6:.3f}M')

    train_set = ChannelDataset(data_root, is_train=1)
    val_set = ChannelDataset(data_root, is_train=0)
    test_set = ChannelDataset(data_root, is_train=2)
    tr = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    va = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    te = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-4)
    crit_mse = nn.MSELoss()
    crit_acc = ACCLoss().to(device)

    best_rate = -1e9
    best_path = os.path.join(output_dir, 'best.pth')
    log_path = os.path.join(output_dir, 'train_log.csv')
    with open(log_path, 'w') as f:
        f.write('epoch,train_loss,val_rate,val_acc,time_sec,saved\n')

    for ep in range(args.epochs):
        model.train()
        t0 = time.time()
        losses = []
        for data in tr:
            opt.zero_grad()
            H0 = data['H'].to(device)
            cl = data['cl'].to(device)
            V, H_rate, clh, p, A = model(H0, cl)
            loss_pre, _, _ = amix_rate_loss(V, H_rate, args.rmin)
            loss = loss_pre + args.gamma2 * crit_mse(clh, cl.squeeze(-1))
            loss.backward()
            opt.step()
            losses.append(loss.item())
        train_loss = float(np.mean(losses))
        vr, va_ = evaluate(model, va, crit_acc)
        saved = vr > best_rate
        if saved:
            best_rate = vr
            torch.save(model.state_dict(), best_path)
        dt = time.time() - t0
        with open(log_path, 'a') as f:
            f.write(f'{ep+1},{train_loss:.7f},{vr:.7f},{va_:.7f},{dt:.1f},{saved}\n')
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f'  ep{ep+1}/{args.epochs} val_rate={vr:.4f} acc={va_:.4f} t={dt:.0f}s {"SAVED" if saved else ""}')

    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    te_r, te_a = evaluate(model, te, crit_acc)
    print(f'TEST rate={te_r:.4f} acc={te_a:.4f}')
    with open(os.path.join(output_dir, 'eval_result.txt'), 'w') as f:
        f.write(f'model={args.model}\ngamma={args.gamma}\nrmin={args.rmin}\n')
        f.write(f'params={nparam}\nTest Rate: {te_r:.4f}\nTest Accuracy: {te_a:.4f}\n')
    print(f'Done: {output_dir}')


if __name__ == '__main__':
    main()
