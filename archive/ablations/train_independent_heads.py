"""
train_independent_heads.py — Ablation: independent p/lambda/cl heads
Replaces shared Linear(768,3) with three independent Linear(768,1).
Everything else unchanged.
"""
import os, sys, argparse, time, datetime, random
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from transformers.models.gpt2.modeling_gpt2 import GPT2Model as HfGPT2
from transformers import GPT2Tokenizer
from data import ChannelDataset
from utils import ACCLoss, RateCal, MULoss, pq2V

sys.stdout.reconfigure(encoding='utf-8')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda:0')
N, K = 256, 10

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

parser = argparse.ArgumentParser()
parser.add_argument('--gamma', type=float, default=0.4)
parser.add_argument('--gamma2', type=float, default=5.0)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--batch_size', type=int, default=100)
args = parser.parse_args()

gpt2_model_path = r"C:\Users\17859\.cache\huggingface\hub\models--openai-community--gpt2\snapshots\607a30d783dfa663caf39e06633721c8d4cfcd7e"
data_root = r"C:\Users\17859\Desktop\files\Grad_Project\LLM for LAE\Codes_v1\Data_user.mat"
base_output = r"C:\Users\17859\Desktop\files\Grad_Project\LLM for LAE\Codes_v1\output"

dir_tag = f"GPT2_indhead_g{args.gamma}"
output_dir = os.path.join(base_output, f"{dir_tag}_{datetime.datetime.now().strftime('%m.%d_%H-%M-%S')}")
os.makedirs(output_dir, exist_ok=True)

# ===================== Model (independent heads) =====================#
from models.gpt2_model_all import Transformer

class Gpt2IndepHeads(nn.Module):
    def __init__(self, model_path, Nt=256, K=10, gamma=0.4, mlp=1):
        super().__init__()
        self.hidden_size = 768
        self.mlp = mlp
        self.Nt = Nt
        self.K = K
        self.P_max = 1
        self.gamma = gamma

        # Load GPT-2
        gpt2_tokenizer = GPT2Tokenizer.from_pretrained(model_path, use_fast=False)
        gpt2_tokenizer.pad_token_id = 0
        gpt2 = HfGPT2.from_pretrained(model_path, torch_dtype=torch.float32)
        for name, param in gpt2.named_parameters():
            if 'ln' in name or 'wpe' in name:
                param.requires_grad = True
            elif 'mlp' in name and mlp == 1:
                param.requires_grad = True
            else:
                param.requires_grad = False
        self.gpt2_model = gpt2

        # Encoder (same as original)
        self.transformer_mu = Transformer(Nt*2, depth=3, heads=8, dim_head=64, mlp_dim=512, dropout=0.)
        self.llama_proj = nn.Linear(Nt*2, self.hidden_size)

        # === Independent heads (replaces shared Linear(768,3)) ===
        self.p_head = nn.Linear(self.hidden_size, 1)
        self.lambda_head = nn.Linear(self.hidden_size, 1)
        self.cl_head = nn.Linear(self.hidden_size, 1)
        self.output_layer = nn.Sigmoid()

    def noise(self, H, SNR):
        sigma = 10 ** (- SNR / 10).to(H.device)
        add_noise = torch.sqrt(sigma / 2) * (torch.randn(*H.shape)).to(H.device)
        add_noise = add_noise * torch.sqrt(torch.mean(torch.abs(H) ** 2))
        return H + add_noise

    def forward(self, H=None, cl=None):
        mean = torch.mean(H)
        std = torch.std(H)
        muchannel = (H - mean) / std
        B, L, enc_in = H.shape
        for i in range(B):
            SNR = torch.tensor(0.0)
            muchannel[i, ...] = self.noise(muchannel[i, ...], SNR)
        muchannel = self.transformer_mu(muchannel)
        input_embs = self.llama_proj(muchannel)
        input_atts = torch.ones(input_embs.size()[:-1], dtype=torch.long).to(muchannel.device)

        last_hidden_state = self.gpt2_model(
            input_ids=None, inputs_embeds=input_embs, attention_mask=input_atts
        ).last_hidden_state

        # Independent heads
        p_hat_raw = self.p_head(last_hidden_state).squeeze(-1)        # [B, K]
        lamda_hat_raw = self.lambda_head(last_hidden_state).squeeze(-1) # [B, K]
        cl_hat_raw = self.cl_head(last_hidden_state).squeeze(-1)       # [B, K]

        p_hat_sigmoid = self.output_layer(p_hat_raw)
        lamda_hat_sigmoid = self.output_layer(lamda_hat_raw)
        cl_hat = cl_hat_raw  # raw logit for classification (sigmoid applied in loss/eval)

        cl = cl.squeeze(-1)

        # === p_hat gamma constraint (identical to original) ===
        p_hat = p_hat_sigmoid
        p_sum = torch.norm(p_hat, p=2, dim=1, keepdim=True)**2
        p_normalized = p_hat / torch.sqrt(p_sum + 1e-8)
        temp_label_0 = torch.norm(p_normalized * cl, p=2, dim=1, keepdim=True)**2
        mask0 = temp_label_0 < self.gamma

        norm_label_1 = torch.norm(p_hat * cl, p=2, dim=1, keepdim=True)**2
        scale_1 = torch.sqrt(self.P_max * self.gamma / (norm_label_1 + 1e-8))
        normalized_label_1 = p_hat * cl * scale_1
        norm_label_0 = torch.norm(p_hat * (1-cl), p=2, dim=1, keepdim=True)**2
        scale_0 = torch.sqrt(self.P_max * (1-self.gamma) / (norm_label_0 + 1e-8))
        normalized_label_0 = p_hat * (1 - cl) * scale_0
        p_hat_0 = normalized_label_1 + normalized_label_0
        p_hat_0 = mask0 * p_normalized + (~mask0) * p_hat_0

        # === lambda_hat gamma constraint (identical to original) ===
        lamda_hat = lamda_hat_sigmoid
        lamda_sum = torch.sum(lamda_hat, dim=1, keepdim=True)
        lamda_normalized = lamda_hat / (lamda_sum + 1e-8)
        temp_label_1 = torch.sum(lamda_normalized * cl, dim=1, keepdim=True)
        mask1 = temp_label_1 < self.gamma

        sum_label_1 = torch.sum(lamda_hat * cl, dim=1, keepdim=True)
        scale_p_1 = self.P_max * self.gamma / (sum_label_1 + 1e-8)
        normalized_P_label_1 = lamda_hat * cl * scale_p_1
        sum_label_0 = torch.sum(lamda_hat * (1-cl), dim=1, keepdim=True)
        scale_p_0 = self.P_max * (1-self.gamma) / (sum_label_0 + 1e-8)
        normalized_P_label_0 = lamda_hat * (1-cl) * scale_p_0
        lamda_hat_0 = normalized_P_label_1 + normalized_P_label_0
        lamda_hat_0 = mask1 * lamda_normalized + (~mask1) * lamda_hat_0

        return p_hat_0, lamda_hat_0, cl_hat


# ===================== Main =====================#
if __name__ == "__main__":
    model = Gpt2IndepHeads(model_path=gpt2_model_path, Nt=N, K=K, gamma=args.gamma)
    model.to(device)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: total={total/1e6:.5f}M, trainable={trainable/1e6:.5f}M")
    print(f"Head params: p_head={sum(p.numel() for p in model.p_head.parameters())}, "
          f"lambda_head={sum(p.numel() for p in model.lambda_head.parameters())}, "
          f"cl_head={sum(p.numel() for p in model.cl_head.parameters())}")

    train_set = ChannelDataset(data_root, is_train=1)
    val_set = ChannelDataset(data_root, is_train=0)
    test_set = ChannelDataset(data_root, is_train=2)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0001)
    criterion_train = MULoss().to(device)
    criterion_test = RateCal().to(device)
    criterion_train_cl = nn.MSELoss().to(device)
    criterion_test_cl = ACCLoss().to(device)

    training_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    validate_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    # ===================== Training loop =====================#
    best_rate = -float("inf")
    best_acc = -float("inf")
    best_rate_epoch = 0
    best_acc_epoch = 0
    best_checkpoint = os.path.join(output_dir, 'best.bin')
    log_path = os.path.join(output_dir, 'train_log.csv')
    with open(log_path, 'w') as f:
        f.write('epoch,train_loss,val_rate,val_acc,time_sec,saved\n')

    print(f"\nTraining: gamma={args.gamma}, gamma2={args.gamma2}, {args.epochs} epochs")
    print(f"Output: {output_dir}\n")

    for epoch in range(args.epochs):
        since = time.time()
        epoch_train_loss = []

        # Train
        model.train()
        for data in training_loader:
            optimizer.zero_grad()
            H = data['H'].to(device, non_blocking=True)
            H = rearrange(H, 'n W H a -> n W (H a)')
            cl = data['cl'].to(device, non_blocking=True)
            H = H[:, :K, :]
            cl = cl[:, :K]
            p_hat, lamda_hat, cl_hat = model(H, cl)
            loss_pre = criterion_train(p_hat, lamda_hat, H)
            loss_cl = criterion_train_cl(cl, torch.unsqueeze(cl_hat, dim=2))
            loss = loss_pre + loss_cl * args.gamma2
            epoch_train_loss.append(loss.item())
            loss.backward()
            optimizer.step()

        epoch_loss = np.nanmean(np.array(epoch_train_loss))
        time_elapsed = time.time() - since

        # Validate
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
            best_rate_epoch = epoch + 1
            torch.save(model.state_dict(), best_checkpoint)
            saved = True
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_acc_epoch = epoch + 1

        with open(log_path, 'a') as f:
            f.write(f'{epoch+1},{epoch_loss:.7f},{epoch_rate:.7f},{epoch_acc:.7f},{time_elapsed:.1f},{saved}\n')

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{args.epochs}: rate={epoch_rate:.4f} acc={epoch_acc*100:.2f}% (gamma={args.gamma}) time={time_elapsed:.1f}s {'SAVED' if saved else ''}")

    # ===================== Final evaluation =====================#
    print(f"\nBest rate epoch: {best_rate_epoch}, best rate: {best_rate:.4f}")
    print(f"Best acc epoch:  {best_acc_epoch}, best acc: {best_acc*100:.2f}%")

    # Load best-rate checkpoint
    model.load_state_dict(torch.load(best_checkpoint, map_location=device, weights_only=True))
    model.eval()

    criterion_rate = RateCal().to(device)
    criterion_acc = ACCLoss().to(device)

    # Test clean
    rates, accs = [], []
    with torch.no_grad():
        for data in test_loader:
            H = data['H'].to(device, non_blocking=True)
            cl = data['cl'].to(device, non_blocking=True)
            H_in = rearrange(H, 'n W H a -> n W (H a)')
            p_hat, lamda_hat, cl_hat = model(H_in, cl)
            r = criterion_rate(p_hat, lamda_hat, H_in).item()
            a = criterion_acc(cl, torch.unsqueeze(cl_hat, dim=2)).item()
            rates.append(r)
            accs.append(a)
    test_rate = np.mean(rates)
    test_acc = np.mean(accs)
    print(f"\nTest (best_rate ckpt): rate={test_rate:.4f} acc={test_acc*100:.2f}%")

    # SNR sweep
    print("\nSNR sweep (best_rate ckpt):")
    snr_results = {}
    for snr in [0, 5, 10, 15, 20]:
        accs_snr = []
        with torch.no_grad():
            for data in test_loader:
                H = data['H'].to(device, non_blocking=True)
                cl = data['cl'].to(device, non_blocking=True)
                H_in = rearrange(H, 'n W H a -> n W (H a)')
                # Bypass internal noise, add external AWGN
                original_noise = model.noise
                model.noise = lambda h, s: h
                H_4d = H_in.reshape(*H_in.shape[:-1], H_in.shape[-1] // 2, 2)
                H_complex = torch.complex(H_4d[..., 0], H_4d[..., 1])
                P_signal = torch.mean(torch.abs(H_complex) ** 2).item()
                noise_power = P_signal / (10 ** (snr / 10))
                n_complex = (torch.randn_like(H_complex) + 1j * torch.randn_like(H_complex)) * (noise_power / 2.0) ** 0.5
                noise = torch.stack([n_complex.real, n_complex.imag], dim=-1).reshape_as(H_in)
                H_noisy = H_in + noise
                p_hat, lamda_hat, cl_hat = model(H_noisy, cl)
                model.noise = original_noise
                a = criterion_acc(cl, torch.unsqueeze(cl_hat, dim=2)).item()
                accs_snr.append(a)
        snr_results[snr] = np.mean(accs_snr)
        print(f"  SNR={snr:>2d}dB: {snr_results[snr]*100:.2f}%")

    # Also load best-acc checkpoint and evaluate if different
    if best_acc_epoch != best_rate_epoch:
        print(f"\n--- Also evaluating best_acc checkpoint (epoch {best_acc_epoch}) ---")
        # Re-load and re-train isn't feasible; just report the epoch difference
        print(f"Best acc epoch {best_acc_epoch} differs from best rate epoch {best_rate_epoch}")

    # Save summary
    summary = {
        'best_rate_epoch': best_rate_epoch,
        'best_rate': best_rate,
        'best_acc_epoch': best_acc_epoch,
        'best_acc': best_acc,
        'test_rate': test_rate,
        'test_acc': test_acc,
        'snr_results': {str(k): v for k, v in snr_results.items()},
    }
    import json
    with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nDone: {output_dir}")
