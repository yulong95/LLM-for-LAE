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
parser.add_argument('--gamma', type=float, default=0.4, help='Near-field power ratio constraint (alpha_c). Paper Fig.6/8/9 uses 0.4')
parser.add_argument('--gamma2', type=float, default=5.0, help='Classification loss weight')
parser.add_argument('--rmin', type=float, default=0.6, help='Minimum rate constraint (Rmin). Paper Fig.8 sweeps this.')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--csi_snr', type=float, default=None,
                    help='Train-time CSI SNR dB (Table I protocol). None=pure CSI (default).')
parser.add_argument('--sup_wmmse', action='store_true',
                    help='[31] supervised: MSE to WMMSE (p,λ) labels + cl; 不用 MULoss')
parser.add_argument('--wmmse_labels', type=str,
                    default=r"C:\Users\17859\Desktop\files\Grad_Project\LLM for LAE\Codes_v1\output\wmmse_labels.pt")
parser.add_argument('--finetune_from', type=str, default=None,
                    help='[31] stage-2: load ckpt and fine-tune with MULoss (rate) + cl')
args = parser.parse_args()


def add_csi_awgn(H_flat, snr_db):
    """H_flat: [..., 2N] interleaved or last-dim even real/imag pairs."""
    H_4d = H_flat.reshape(*H_flat.shape[:-1], H_flat.shape[-1] // 2, 2)
    H_complex = torch.complex(H_4d[..., 0], H_4d[..., 1])
    p_sig = torch.mean(torch.abs(H_complex) ** 2)
    noise_cplx_power = p_sig / (10 ** (snr_db / 10.0))
    n_complex = (torch.randn_like(H_complex) + 1j * torch.randn_like(H_complex)) * torch.sqrt(noise_cplx_power / 2.0)
    noise = torch.stack([n_complex.real, n_complex.imag], dim=-1).reshape_as(H_flat)
    return H_flat + noise

# ============= Paths =============#
data_root = r"C:\Users\17859\Desktop\files\Grad_Project\LLM for LAE\Codes_v1\Data_user.mat"
base_output = r"C:\Users\17859\Desktop\files\Grad_Project\LLM for LAE\Codes_v1\output"

# Output dir naming (paper default: gamma=0.4, gamma2=5.0, rmin=0.6)
if args.finetune_from:
    dir_tag = "CNN_ftwmmse"
elif args.sup_wmmse:
    dir_tag = "CNN_supwmmse"
elif args.csi_snr is not None:
    dir_tag = f"CNN_csi{args.csi_snr:.0f}"
elif args.gamma != 0.4:
    dir_tag = f"CNN_gamma{args.gamma:.1f}"
elif args.gamma2 != 5.0:
    dir_tag = f"CNN_gamma2_{args.gamma2}"
elif args.rmin != 0.6:
    dir_tag = f"CNN_rmin{args.rmin:.1f}"
else:
    dir_tag = "CNN"
output_dir = os.path.join(base_output, f"{dir_tag}_{datetime.datetime.now().strftime('%m.%d_%H-%M-%S')}")
os.makedirs(output_dir, exist_ok=True)

# ============= Model =============#
model = CNN_pre(N, K, args.gamma)
model.to(device)

# Stage-2 fine-tune from supervised ckpt ([30] phase-2 / [31] 两阶段)
if args.finetune_from:
    ck = args.finetune_from
    if os.path.isdir(ck):
        cands = sorted(glob.glob(os.path.join(ck, '*.pth')), key=os.path.getmtime)
        if not cands:
            raise FileNotFoundError(f'no .pth in {ck}')
        ck = cands[-1]
    model.load_state_dict(torch.load(ck, map_location=device, weights_only=True))
    print(f"Fine-tune init from {ck}")
    # stage-2: MULoss，不用 WMMSE 标签
    args.sup_wmmse = False

# [31] supervised labels (optional, stage-1)
WMMSE_LAB = None
if args.sup_wmmse:
    WMMSE_LAB = torch.load(args.wmmse_labels, map_location='cpu')
    print(f"Loaded WMMSE labels: {list(WMMSE_LAB.keys())} train_p={tuple(WMMSE_LAB['train']['p'].shape)}")


class IndexedDataset(torch.utils.data.Dataset):
    def __init__(self, ds):
        self.ds = ds

    def __getitem__(self, i):
        d = self.ds[i]
        d['index'] = i
        return d

    def __len__(self):
        return len(self.ds)


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
            if args.csi_snr is not None:
                Hf = H.reshape(H.shape[0], H.shape[1], -1)
                Hf = add_csi_awgn(Hf, args.csi_snr)
                H = Hf.reshape_as(H)
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
            if args.sup_wmmse:
                # [31] supervised: MSE to WMMSE (p,λ) + classification
                idx = data['index']
                p_lab = WMMSE_LAB['train']['p'][idx].to(device)
                lam_lab = WMMSE_LAB['train']['lam'][idx].to(device)
                loss_pre = criterion_train_cl(p_hat, p_lab) + criterion_train_cl(lamda_hat, lam_lab)
            else:
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
    if args.sup_wmmse:
        train_set = IndexedDataset(train_set)
        validate_set = IndexedDataset(validate_set)
        test_set = IndexedDataset(test_set)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0001)
    criterion_train = MULoss(rmin=args.rmin).to(device)
    criterion_test = RateCal().to(device)
    criterion_train_cl = nn.MSELoss().to(device)
    criterion_test_cl = ACCLoss().to(device)

    training_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True)
    validate_loader = torch.utils.data.DataLoader(dataset=validate_set, batch_size=args.batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=False)

    train(training_loader, validate_loader, test_loader)
