"""
CNN.py — CNN baseline training (supports --gamma)
Usage:
  python CNN.py                    # default: gamma=0.8
  python CNN.py --gamma 0.5        # sweep gamma
"""
import os, sys, argparse, time, datetime, glob
import torch
import torch.nn as nn
import numpy as np
from models.baseline_CNN import CNN_pre
from data import ChannelDataset
from utils import ACCLoss, RateCal, MULoss
from einops import rearrange

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda:0')
N, K = 256, 10

# ============= Args =============#
parser = argparse.ArgumentParser()
parser.add_argument('--gamma', type=float, default=0.8, help='Near-field power ratio constraint')
parser.add_argument('--gamma2', type=float, default=5.0, help='Classification loss weight')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--batch_size', type=int, default=100)
args = parser.parse_args()

# ============= Paths =============#
data_root = r"C:\Users\17859\Desktop\files\Grad_Project\LLM for LAE\Codes_v1\Data_user.mat"
base_output = r"C:\Users\17859\Desktop\files\Grad_Project\LLM for LAE\Codes_v1\output"

# Output dir naming
if args.gamma != 0.8:
    dir_tag = f"CNN_gamma{args.gamma:.1f}"
else:
    dir_tag = "CNN"
output_dir = os.path.join(base_output, f"{dir_tag}_{datetime.datetime.now().strftime('%m.%d_%H-%M-%S')}")
os.makedirs(output_dir, exist_ok=True)

# ============= Model =============#
model = CNN_pre(N, K, args.gamma)
model.to(device)


# ============= Train function =============#
def train(training_loader, validate_loader, test_loader):
    best_loss = 0
    log_path = os.path.join(output_dir, 'train_log_cnn.csv')
    with open(log_path, 'w') as f:
        f.write('epoch,train_loss,val_loss,val_acc,time_sec,saved\n')

    def cleanup_checkpoints(keep_best=2):
        ckpts = sorted(glob.glob(os.path.join(output_dir, '*.pth')), key=lambda x: int(os.path.basename(x).split('.')[0]))
        if len(ckpts) > keep_best:
            for f in ckpts[:-keep_best]:
                os.remove(f)

    print(f"Training CNN: gamma={args.gamma}, gamma2={args.gamma2}, {args.epochs} epochs")
    print(f"Output: {output_dir}")

    for epoch in range(args.epochs):
        since = time.time()
        epoch_train_loss = []

        # Train
        model.train()
        for data in training_loader:
            optimizer.zero_grad()
            H = data['H'].to(device, non_blocking=True)
            H = rearrange(H, 'n k W H -> n H W k', H=2)
            cl = data['cl'].to(device, non_blocking=True)
            random_int = 10
            H0 = torch.zeros(H.shape, device=device)
            cl0 = torch.zeros(cl.shape, device=device)
            H = H[:, :, :, :random_int]
            cl = cl[:, :random_int]
            mean = torch.mean(H)
            std = torch.std(H)
            H0[:, :, :, :random_int] = H
            cl0[:, :random_int] = cl

            p_hat, lamda_hat, cl_hat = model(H0, cl, random_int, mean, std)
            H = rearrange(H, 'n H W k -> n k (W H)')
            loss_pre = criterion_train(p_hat, lamda_hat, H)
            loss_cl = criterion_train_cl(cl, torch.unsqueeze(cl_hat, dim=2))
            loss = loss_pre + loss_cl * args.gamma2
            epoch_train_loss.append(loss.item())
            loss.backward()
            optimizer.step()

        epoch_loss = np.nanmean(np.array(epoch_train_loss))
        train_loss_val = epoch_loss
        time_elapsed = time.time() - since

        # Validate
        model.eval()
        epoch_val_loss = []
        epoch_val_loss_cl = []
        with torch.no_grad():
            for data in validate_loader:
                H = data['H'].to(device, non_blocking=True)
                H = rearrange(H, 'n k W H -> n H W k', H=2)
                cl = data['cl'].to(device, non_blocking=True)
                random_int = 10
                H0 = torch.zeros(H.shape, device=device)
                cl0 = torch.zeros(cl.shape, device=device)
                H = H[:, :, :, :random_int]
                cl = cl[:, :random_int]
                mean = torch.mean(H)
                std = torch.std(H)
                H0[:, :, :, :random_int] = H
                cl0[:, :random_int] = cl

                p_hat, lamda_hat, cl_hat = model(H0, cl, random_int, mean, std)
                H = rearrange(H, 'n H W k -> n k (W H)')
                loss_pre = criterion_test(p_hat, lamda_hat, H)
                loss_cl = criterion_test_cl(cl, torch.unsqueeze(cl_hat, dim=2))
                epoch_val_loss.append(loss_pre.item())
                epoch_val_loss_cl.append(loss_cl.item())

        epoch_rate = np.nanmean(np.array(epoch_val_loss))
        epoch_acc = np.nanmean(np.array(epoch_val_loss_cl))
        epoch_loss_val = epoch_rate + epoch_acc

        # MULoss for CSV (same as training, for Fig.5)
        epoch_mu_loss = []
        with torch.no_grad():
            for data in validate_loader:
                H = data['H'].to(device, non_blocking=True)
                H = rearrange(H, 'n k W H -> n H W k', H=2)
                cl = data['cl'].to(device, non_blocking=True)
                random_int = 10
                H0 = torch.zeros(H.shape, device=device)
                cl0 = torch.zeros(cl.shape, device=device)
                H = H[:, :, :, :random_int]
                cl = cl[:, :random_int]
                mean = torch.mean(H)
                std = torch.std(H)
                H0[:, :, :, :random_int] = H
                cl0[:, :random_int] = cl
                p_hat, lamda_hat, cl_hat = model(H0, cl, random_int, mean, std)
                H = rearrange(H, 'n H W k -> n k (W H)')
                mu_loss = criterion_train(p_hat, lamda_hat, H)
                epoch_mu_loss.append(mu_loss.item())
        val_mu_loss = np.nanmean(np.array(epoch_mu_loss))

        saved = False
        if epoch_loss_val > best_loss:
            best_loss = epoch_loss_val
            torch.save(model.state_dict(), os.path.join(output_dir, '%d.pth' % (epoch + 1)))
            cleanup_checkpoints(keep_best=2)
            saved = True

        with open(log_path, 'a') as f:
            f.write(f'{epoch+1},{train_loss_val:.7f},{val_mu_loss:.7f},{epoch_acc:.7f},{time_elapsed:.1f},{saved}\n')

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{args.epochs}: rate={epoch_rate:.4f} acc={epoch_acc:.4f} time={time_elapsed:.1f}s {'SAVED' if saved else ''}")

    # Final test
    print("\nFinal test evaluation...")
    ckpts = sorted([f for f in os.listdir(output_dir) if f.endswith('.pth')], key=lambda x: int(x.split('.')[0]))
    for ck in reversed(ckpts):
        try:
            model.load_state_dict(torch.load(os.path.join(output_dir, ck), map_location=device, weights_only=True))
            break
        except:
            pass
    model.eval()

    criterion_rate = RateCal().to(device)
    criterion_acc = ACCLoss().to(device)
    rates, accs = [], []
    with torch.no_grad():
        for data in test_loader:
            H = data['H'].to(device, non_blocking=True)
            cl = data['cl'].to(device, non_blocking=True)
            H_in = rearrange(H, 'n k W H -> n H W k', H=2)
            H0 = torch.zeros(H_in.shape, device=device)
            H0[:, :, :, :K] = H_in[:, :, :, :K]
            mean = torch.mean(H_in[:, :, :, :K])
            std = torch.std(H_in[:, :, :, :K])
            p_hat, lamda_hat, cl_hat = model(H0, cl, K, mean, std)
            H_rate = rearrange(H0, 'n H W k -> n k (W H)')
            r = criterion_rate(p_hat, lamda_hat, H_rate).item()
            a = criterion_acc(cl, torch.unsqueeze(cl_hat, dim=2)).item()
            rates.append(r)
            accs.append(a)

    avg_rate = np.mean(rates)
    avg_acc = np.mean(accs)
    print(f"Test: rate={avg_rate:.4f} acc={avg_acc:.4f}")

    with open(os.path.join(output_dir, 'eval_result.txt'), 'w') as f:
        f.write(f'gamma={args.gamma}\ngamma2={args.gamma2}\n')
        f.write(f'Test Rate: {avg_rate:.4f}\nTest Accuracy: {avg_acc:.4f}\n')

    print(f"Done: {output_dir}")


# ============= Main =============#
if __name__ == "__main__":
    total = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {total/1e6:.5f}M")

    train_set = ChannelDataset(data_root, is_train=1)
    validate_set = ChannelDataset(data_root, is_train=0)
    test_set = ChannelDataset(data_root, is_train=2)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0001)
    criterion_train = MULoss().to(device)
    criterion_test = RateCal().to(device)
    criterion_train_cl = nn.MSELoss().to(device)
    criterion_test_cl = ACCLoss().to(device)

    training_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True)
    validate_loader = torch.utils.data.DataLoader(dataset=validate_set, batch_size=args.batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=False)

    train(training_loader, validate_loader, test_loader)
