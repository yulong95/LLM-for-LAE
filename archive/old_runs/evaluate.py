import os
import sys
import glob
import torch
import torch.nn as nn
import numpy as np
from models.gpt2_model_all import Gpt2Model
from models.baseline_CNN import CNN_pre
from data import ChannelDataset
from utils import ACCLoss, RateCal, MULoss
from einops import rearrange
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ============= CONFIG ==========#
batch_size = 100
device = torch.device('cuda:0')
N = 256
K = 10

gpt2_model_path = r"C:\Users\17859\.cache\huggingface\hub\models--openai-community--gpt2\snapshots\607a30d783dfa663caf39e06633721c8d4cfcd7e"
data_root = r"C:\Users\17859\Desktop\files\Grad_Project\LLM for LAE\Codes_v1\Data_user.mat"
base_output = r"C:\Users\17859\Desktop\files\Grad_Project\LLM for LAE\Codes_v1\output"

# ============= AUTO-FIND LATEST RUN ==========#
def find_latest_run(model_type):
    pattern = os.path.join(base_output, f"{model_type}_*")
    runs = sorted(glob.glob(pattern))
    if not runs:
        print(f"No {model_type} runs found in {base_output}")
        sys.exit(1)
    print(f"Available {model_type} runs:")
    for i, r in enumerate(runs):
        print(f"  [{i}] {os.path.basename(r)}")
    return runs[-1]  # latest

# ============= PARSE ARGS ==========#
# Usage: python evaluate.py [gpt2|cnn] [run_dir]
# Examples:
#   python evaluate.py                  -> latest GPT2 run
#   python evaluate.py cnn              -> latest CNN run
#   python evaluate.py gpt2 GPT2_08.27_15-30-45  -> specific run

MODEL_TYPE = sys.argv[1] if len(sys.argv) > 1 else 'gpt2'
if MODEL_TYPE not in ('gpt2', 'cnn'):
    print(f"Unknown model type: {MODEL_TYPE}, use 'gpt2' or 'cnn'")
    sys.exit(1)

if len(sys.argv) > 2:
    run_dir = os.path.join(base_output, sys.argv[2])
else:
    run_dir = find_latest_run(MODEL_TYPE)

print(f"Evaluating: {os.path.basename(run_dir)}")

# Find checkpoint
ext = '.bin' if MODEL_TYPE == 'gpt2' else '.pth'
ckpts = sorted(glob.glob(os.path.join(run_dir, f'*{ext}')), key=os.path.getmtime)
if not ckpts:
    print(f"No checkpoint found in {run_dir}")
    sys.exit(1)
weight_path = ckpts[-1]
print(f"Using checkpoint: {os.path.basename(weight_path)}")

# ============= BUILD MODEL ==========#
if MODEL_TYPE == 'gpt2':
    gamma = 0.99
    model = Gpt2Model(model_path=gpt2_model_path, Nt=N, K=K, gamma=gamma)
else:
    gamma = 0.8
    model = CNN_pre(N, K, gamma)

model.to(device)
model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
model.eval()

# ============= EVALUATE (on test set) ==========#
validate_set = ChannelDataset(data_root, is_train=2, train_per=0.8, valid_per=0.1)
validate_loader = torch.utils.data.DataLoader(dataset=validate_set, batch_size=batch_size, shuffle=False)
criterion_rate = RateCal().to(device)
criterion_acc = ACCLoss().to(device)

epoch_val_loss = []
epoch_val_loss_cl = []

with torch.no_grad():
    for iteration, data in enumerate(validate_loader):
        H = data['H'].to(device, non_blocking=True)
        cl = data['cl'].to(device, non_blocking=True)

        if MODEL_TYPE == 'gpt2':
            H_input = rearrange(H, 'n W H a -> n W (H a)')
            p_hat, lamda_hat, cl_hat = model(H_input, cl)
            H_rate = H_input
        else:
            H_input = rearrange(H, 'n k W H -> n H W k', H=2)
            random_int = 10
            mean = torch.mean(H_input)
            std = torch.std(H_input)
            p_hat, lamda_hat, cl_hat = model(H_input, cl, random_int, mean, std)
            H_rate = rearrange(H, 'n k W H -> n k (W H)')

        loss_rate = criterion_rate(p_hat, lamda_hat, H_rate)
        loss_acc = criterion_acc(cl, torch.unsqueeze(cl_hat, dim=2))
        epoch_val_loss.append(loss_rate.item())
        epoch_val_loss_cl.append(loss_acc.item())

avg_rate = np.nanmean(np.array(epoch_val_loss))
avg_acc = np.nanmean(np.array(epoch_val_loss_cl))
print(f'Sum-Rate: {avg_rate:.7f} bits/s/Hz')
print(f'Classification Accuracy: {avg_acc:.7f}')

# Save result
result_path = os.path.join(run_dir, 'eval_result.txt')
with open(result_path, 'w') as f:
    f.write(f'Checkpoint: {os.path.basename(weight_path)}\n')
    f.write(f'Sum-Rate: {avg_rate:.7f} bits/s/Hz\n')
    f.write(f'Classification Accuracy: {avg_acc:.7f}\n')
print(f'Result saved to {result_path}')
