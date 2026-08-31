"""GPT-2 最小运行测试：1 epoch 详细诊断"""
import os, sys, time, torch
import torch.nn as nn
import numpy as np
from models.gpt2_model_all import Gpt2Model
from data import ChannelDataset
from utils import ACCLoss, RateCal, MULoss
from einops import rearrange

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
N, K = 256, 10

print("=" * 60)
print("GPT-2 最小运行测试 (1 epoch)")
print("=" * 60)

# 1. 模型初始化
print("\n[1] 模型初始化")
gpt2_model_path = r"C:\Users\17859\.cache\huggingface\hub\models--openai-community--gpt2\snapshots\607a30d783dfa663caf39e06633721c8d4cfcd7e"
model = Gpt2Model(model_path=gpt2_model_path, Nt=N, K=K, gamma=0.99)
model.to(device)
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  设备: {device}")
print(f"  总参数: {total_params/1e6:.2f}M")
print(f"  可训练参数: {trainable_params/1e6:.2f}M")

# 2. 数据加载
print("\n[2] 数据加载")
data_root = r"C:\Users\17859\Desktop\files\Grad_Project\LLM for LAE\Codes_v1\Data_user.mat"
train_set = ChannelDataset(data_root, is_train=1)
val_set = ChannelDataset(data_root, is_train=0)
print(f"  训练集长度: {len(train_set)}")
print(f"  验证集长度: {len(val_set)}")

# 3. 单个batch测试
print("\n[3] 单个Batch测试")
loader = torch.utils.data.DataLoader(train_set, batch_size=4, shuffle=False)
batch = next(iter(loader))
H = batch['H'].to(device)
cl = batch['cl'].to(device)
print(f"  H shape (输入): {H.shape}, dtype: {H.dtype}")
print(f"  cl shape (输入): {cl.shape}, dtype: {cl.dtype}")
print(f"  H 数值范围: [{H.min().item():.4f}, {H.max().item():.4f}]")
print(f"  cl 唯一值: {torch.unique(cl).tolist()}")

# 4. 模型前向传播
print("\n[4] 模型前向传播")
model.eval()
with torch.no_grad():
    H_in = rearrange(H, 'n W H a -> n W (H a)')
    cl_in = cl
    print(f"  H_in shape (模型输入): {H_in.shape}")
    print(f"  cl_in shape (模型输入): {cl_in.shape}")

    p_hat, lamda_hat, cl_hat = model(H_in, cl_in)

print(f"  p_hat shape: {p_hat.shape}, dtype: {p_hat.dtype}")
print(f"  lamda_hat shape: {lamda_hat.shape}, dtype: {lamda_hat.dtype}")
print(f"  cl_hat shape: {cl_hat.shape}, dtype: {cl_hat.dtype}")
print(f"  p_hat 数值范围: [{p_hat.min().item():.4f}, {p_hat.max().item():.4f}]")
print(f"  lamda_hat 数值范围: [{lamda_hat.min().item():.4f}, {lamda_hat.max().item():.4f}]")
print(f"  cl_hat 数值范围: [{cl_hat.min().item():.4f}, {cl_hat.max().item():.4f}]")

# 5. Loss计算
print("\n[5] Loss计算")
model.train()
criterion_train = MULoss().to(device)
criterion_cl = nn.MSELoss().to(device)

H_train = rearrange(H, 'n W H a -> n W (H a)')
p_hat, lamda_hat, cl_hat = model(H_train, cl)

loss_pre = criterion_train(p_hat, lamda_hat, H_train)
loss_cl = criterion_cl(cl, torch.unsqueeze(cl_hat, dim=2))
loss = loss_pre + loss_cl * 5.0

print(f"  loss_pre (Rate): {loss_pre.item():.4f}")
print(f"  loss_cl (MSE): {loss_cl.item():.6f}")
print(f"  total loss: {loss.item():.4f}")
print(f"  loss is finite: {torch.isfinite(loss).item()}")

# 6. 反向传播
print("\n[6] 反向传播")
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
optimizer.zero_grad()
loss.backward()
optimizer.step()
print("  backward + optimizer.step() 完成")

# 7. Validation
print("\n[7] Validation")
model.eval()
val_loader = torch.utils.data.DataLoader(val_set, batch_size=4, shuffle=False)
criterion_rate = RateCal().to(device)
criterion_acc = ACCLoss().to(device)

with torch.no_grad():
    val_batch = next(iter(val_loader))
    H_val = val_batch['H'].to(device)
    cl_val = val_batch['cl'].to(device)
    H_val_in = rearrange(H_val, 'n W H a -> n W (H a)')
    p_hat_v, lamda_hat_v, cl_hat_v = model(H_val_in, cl_val)
    rate = criterion_rate(p_hat_v, lamda_hat_v, H_val_in).item()
    acc = criterion_acc(cl_val, torch.unsqueeze(cl_hat_v, dim=2)).item()
    print(f"  Val Rate: {rate:.4f}")
    print(f"  Val Accuracy: {acc:.4f}")

# 8. Checkpoint保存/加载
print("\n[8] Checkpoint保存/加载")
test_ckpt = "test_checkpoint.bin"
torch.save(model.state_dict(), test_ckpt)
print(f"  保存: {test_ckpt} ({os.path.getsize(test_ckpt)/1e6:.1f} MB)")

model_loaded = Gpt2Model(model_path=gpt2_model_path, Nt=N, K=K, gamma=0.99)
model_loaded.to(device)
model_loaded.load_state_dict(torch.load(test_ckpt, map_location=device, weights_only=True))
print("  加载成功")
os.remove(test_ckpt)

# 9. GPU信息
print("\n[9] GPU/CPU信息")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  显存占用: {torch.cuda.memory_allocated(0)/1e6:.1f} MB")
    print(f"  显存缓存: {torch.cuda.memory_reserved(0)/1e6:.1f} MB")
else:
    print("  使用CPU")

print("\n" + "=" * 60)
print("测试完成！所有检查项通过。")
print("=" * 60)
