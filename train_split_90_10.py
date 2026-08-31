"""
train_split_90_10.py — 作者式90/10数据划分对照实验
与当前80/10/10划分对比，验证数据划分是否解释95% vs 99%差距

唯一改动：数据划分为90%训练 + 10%验证（无独立测试集）
其他所有参数与 train_randomK.py 完全一致：
  - 训练随机K=3~10
  - 验证/评估固定K=10
  - gamma=0.4, gamma2=5.0
  - 100 epochs
"""
import os, sys, argparse, time, datetime
import torch
import torch.nn as nn
import numpy as np
from models.gpt2_model_all import Gpt2Model
from data import ChannelDataset
from utils import ACCLoss, RateCal, MULoss, pq2V
from einops import rearrange

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda:0')
N, K = 256, 10

parser = argparse.ArgumentParser()
parser.add_argument('--gamma', type=float, default=0.4)
parser.add_argument('--gamma2', type=float, default=5.0)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--batch_size', type=int, default=100)
args = parser.parse_args()

gpt2_model_path = r"C:\Users\17859\.cache\huggingface\hub\models--openai-community--gpt2\snapshots\607a30d783dfa663caf39e06633721c8d4cfcd7e"
data_root = r"C:\Users\17859\Desktop\files\Grad_Project\LLM for LAE\Codes_v1\Data_user.mat"
base_output = r"C:\Users\17859\Desktop\files\Grad_Project\LLM for LAE\Codes_v1\output"

dir_tag = f"GPT2_split90_10_randomK_gamma{args.gamma}"
output_dir = os.path.join(base_output, f"{dir_tag}_{datetime.datetime.now().strftime('%m.%d_%H-%M-%S')}")
os.makedirs(output_dir, exist_ok=True)

model = Gpt2Model(model_path=gpt2_model_path, Nt=N, K=K, gamma=args.gamma)
model.to(device)


def train(training_loader, validate_loader):
    best_rate = -float("inf")
    best_acc = -float("inf")
    best_epoch = 0
    best_checkpoint = os.path.join(output_dir, 'best.bin')
    log_path = os.path.join(output_dir, 'train_log.csv')
    with open(log_path, 'w') as f:
        f.write('epoch,train_loss,val_rate,val_acc,time_sec,saved\n')

    print(f"Training: 90/10 split, random K∈{{3..10}}, gamma={args.gamma}, {args.epochs} epochs")
    print(f"Output: {output_dir}")

    for epoch in range(args.epochs):
        since = time.time()
        epoch_train_loss = []

        # ========== Train (随机K) ==========
        model.train()
        for data in training_loader:
            optimizer.zero_grad()
            H = data['H'].to(device, non_blocking=True)
            H = rearrange(H, 'n W H a -> n W (H a)')
            cl = data['cl'].to(device, non_blocking=True)
            random_int = np.random.randint(8) + 3
            H = H[:, :random_int, :]
            cl = cl[:, :random_int]
            p_hat, lamda_hat, cl_hat = model(H, cl)
            loss_pre = criterion_train(p_hat, lamda_hat, H)
            loss_cl = criterion_train_cl(cl, torch.unsqueeze(cl_hat, dim=2))
            loss = loss_pre + loss_cl * args.gamma2
            epoch_train_loss.append(loss.item())
            loss.backward()
            optimizer.step()

        epoch_loss = np.nanmean(np.array(epoch_train_loss))
        time_elapsed = time.time() - since

        # ========== Validate (固定K=10) ==========
        model.eval()
        epoch_val_loss = []
        epoch_val_loss_cl = []
        with torch.no_grad():
            for data in validate_loader:
                H = data['H'].to(device, non_blocking=True)
                H = rearrange(H, 'n W H a -> n W (H a)')
                cl = data['cl'].to(device, non_blocking=True)
                p_hat, lamda_hat, cl_hat = model(H, cl)
                loss_pre = criterion_test(p_hat, lamda_hat, H)
                loss_cl = criterion_test_cl(cl, torch.unsqueeze(cl_hat, dim=2))
                epoch_val_loss.append(loss_pre.item())
                epoch_val_loss_cl.append(loss_cl.item())

        epoch_rate = np.nanmean(np.array(epoch_val_loss))
        epoch_acc = np.nanmean(np.array(epoch_val_loss_cl))

        saved = False
        if epoch_rate > best_rate:
            best_rate = epoch_rate
            best_epoch = epoch + 1
            torch.save(model.state_dict(), best_checkpoint)
            saved = True
        if epoch_acc > best_acc:
            best_acc = epoch_acc

        with open(log_path, 'a') as f:
            f.write(f'{epoch+1},{epoch_loss:.7f},{epoch_rate:.7f},{epoch_acc:.7f},{time_elapsed:.1f},{saved}\n')

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{args.epochs}: rate={epoch_rate:.4f} acc={epoch_acc*100:.2f}% (K=3~10) time={time_elapsed:.1f}s {'SAVED' if saved else ''}")

    print(f"\nBest epoch: {best_epoch}")
    print(f"Best val rate: {best_rate:.4f}")
    print(f"Best val acc:  {best_acc*100:.2f}%")

    # ========== 在验证集上做详细评估 ==========
    print("\nDetailed validation evaluation (K=10)...")
    model.load_state_dict(torch.load(best_checkpoint, map_location=device, weights_only=True))
    model.eval()

    all_preds, all_targets = [], []
    rates, accs = [], []
    criterion_rate = RateCal().to(device)
    criterion_acc = ACCLoss().to(device)

    with torch.no_grad():
        for data in validate_loader:
            H = data['H'].to(device, non_blocking=True)
            cl = data['cl'].to(device, non_blocking=True)
            H_in = rearrange(H, 'n W H a -> n W (H a)')
            p_hat, lamda_hat, cl_hat = model(H_in, cl)
            r = criterion_rate(p_hat, lamda_hat, H_in).item()
            a = criterion_acc(cl, torch.unsqueeze(cl_hat, dim=2)).item()
            rates.append(r)
            accs.append(a)
            preds = (torch.sigmoid(cl_hat) >= 0.5).float()
            all_preds.append(preds.cpu())
            all_targets.append(cl.squeeze(-1).cpu())

    val_rate = np.mean(rates)
    val_acc = np.mean(accs)
    print(f"Validation (K=10): rate={val_rate:.4f} acc={val_acc*100:.2f}%")

    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    tp = ((preds == 1) & (targets == 1)).sum().item()
    tn = ((preds == 0) & (targets == 0)).sum().item()
    fp = ((preds == 1) & (targets == 0)).sum().item()
    fn = ((preds == 0) & (targets == 1)).sum().item()
    near_total = (targets == 1).sum().item()
    far_total = (targets == 0).sum().item()
    near_acc = tp / near_total if near_total > 0 else 0
    far_acc = tn / far_total if far_total > 0 else 0

    print(f"\n=== Validation Set Detail ===")
    print(f"  TP={tp}, FN={fn}, FP={fp}, TN={tn}")
    print(f"  Near-field accuracy: {near_acc*100:.2f}% ({tp}/{near_total})")
    print(f"  Far-field accuracy:  {far_acc*100:.2f}% ({tn}/{far_total})")

    # Summary
    print(f"\n{'='*50}")
    print(f"90/10 Split Experiment Summary")
    print(f"  Best val rate:  {best_rate:.4f}")
    print(f"  Best val acc:   {best_acc*100:.2f}%")
    print(f"  Val rate (K=10): {val_rate:.4f}")
    print(f"  Val acc (K=10):  {val_acc*100:.2f}%")
    print(f"{'='*50}")

    import json
    summary = {
        'best_epoch': best_epoch,
        'best_val_rate': best_rate,
        'best_val_acc': best_acc,
        'val_rate_k10': val_rate,
        'val_acc_k10': val_acc,
    }
    with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nDone: {output_dir}")


if __name__ == "__main__":
    total = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {total/1e6:.5f}M")

    # 90/10 split: train_per=0.9, valid_per=0.1
    train_set = ChannelDataset(data_root, is_train=1, train_per=0.9, valid_per=0.1)
    validate_set = ChannelDataset(data_root, is_train=0, train_per=0.9, valid_per=0.1)
    print(f"Train: {len(train_set)} samples, Validation: {len(validate_set)} samples")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0001)
    criterion_train = MULoss().to(device)
    criterion_test = RateCal().to(device)
    criterion_train_cl = nn.MSELoss().to(device)
    criterion_test_cl = ACCLoss().to(device)

    training_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True)
    validate_loader = torch.utils.data.DataLoader(dataset=validate_set, batch_size=args.batch_size, shuffle=False)

    train(training_loader, validate_loader)
