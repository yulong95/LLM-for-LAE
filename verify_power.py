"""验证 Fig.9 功率缩放是否正确"""
import os, sys, torch
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from models.gpt2_model_all import Gpt2Model
from data import ChannelDataset
from utils import pq2V, SMR_loss

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda:0')
N, K = 256, 10

print("=" * 60)
print("验证 Fig.9 功率缩放")
print("=" * 60)

# 加载模型
gpt2_model_path = r"C:\Users\17859\.cache\huggingface\hub\models--openai-community--gpt2\snapshots\607a30d783dfa663caf39e06633721c8d4cfcd7e"
data_root = r"C:\Users\17859\Desktop\files\Grad_Project\LLM for LAE\Codes_v1\Data_user.mat"
base_output = r"C:\Users\17859\Desktop\files\Grad_Project\LLM for LAE\Codes_v1\output"

# 找到最新的 GPT2 checkpoint
import glob
runs = sorted(glob.glob(os.path.join(base_output, "GPT2_*")))
valid = [r for r in runs if glob.glob(os.path.join(r, '*.bin')) and 'gamma' not in os.path.basename(r)]
run_dir = valid[-1] if valid else runs[-1]

model = Gpt2Model(model_path=gpt2_model_path, Nt=N, K=K, gamma=0.99)
model.to(device)
ckpts = sorted(glob.glob(os.path.join(run_dir, '*.bin')), key=os.path.getmtime)
for ck in reversed(ckpts):
    try:
        model.load_state_dict(torch.load(ck, map_location=device, weights_only=True))
        break
    except:
        pass
model.eval()
print(f"模型加载: {os.path.basename(run_dir)}")

# 加载一个 batch
test_set = ChannelDataset(data_root, is_train=2)
loader = torch.utils.data.DataLoader(test_set, batch_size=10, shuffle=False)
batch = next(iter(loader))
H = batch['H'].to(device)
cl = batch['cl'].to(device)
H = rearrange(H, 'n W H a -> n W (H a)')

# 测试三个功率点
sigma2 = 10**(-20/10)  # 固定噪声功率 0.01 W
test_points = [
    (-10, 0.1),   # P = -10 dBW = 0.1 W
    (0, 1.0),     # P = 0 dBW = 1 W
    (10, 10.0),   # P = 10 dBW = 10 W
]

print("\n测试功率点:")
print(f"{'P_dBW':>8} {'P_target(W)':>12} {'P_actual(W)':>12} {'误差(%)':>10}")
print("-" * 50)

with torch.no_grad():
    p_hat, lamda_hat, cl_hat = model(H, cl)

    for P_dBW, P_target in test_points:
        # 生成 precoding matrix
        V = pq2V(p_hat, lamda_hat, H, sigma2, N, P_total=P_target)

        # 计算实际总功率
        # V shape: [B, Nt, 1, K, 2] where last dim is real/imag
        V_real = V[..., 0]  # [B, Nt, 1, K]
        V_imag = V[..., 1]  # [B, Nt, 1, K]
        V_power = V_real**2 + V_imag**2  # |V_k|^2
        actual_power = torch.sum(V_power, dim=(1,2,3))  # sum over Nt, 1, K -> [B]
        mean_power = actual_power.mean().item()
        error_pct = abs(mean_power - P_target) / P_target * 100

        print(f"{P_dBW:>8} {P_target:>12.4f} {mean_power:>12.4f} {error_pct:>9.2f}%")

print("\n" + "=" * 60)
print("验证完成")
print("=" * 60)
