import os
import json
import random
import time
import numpy as np
from torch.utils.data import Dataset
import scipy.io as sio
from torchvision import transforms
from einops import rearrange
import torch

class ChannelDataset(Dataset):
    """
    is_train=1: training set   [0, train_per)
    is_train=0: validation set [train_per, train_per+valid_per)
    is_train=2: test set       [train_per+valid_per, 1.0)
    """
    def __init__(self, data_root, is_train=1, train_per=0.8, valid_per=0.1):
        super(ChannelDataset, self).__init__()

        H = sio.loadmat(data_root)['h_near_slant']
        cl = sio.loadmat(data_root)["index_far_near"]
        H = rearrange(H, '(n L) W-> n L W', L=10)
        cl = rearrange(cl, '(n L) W-> n L W', L=10)
        B, L, Nt = H.shape
        if is_train == 1:
            H = H[:int(train_per * B), :, :]
            cl = cl[:int(train_per * B), :, :]
        elif is_train == 0:
            H = H[int(train_per * B):int((train_per + valid_per) * B), :, :]
            cl = cl[int(train_per * B):int((train_per + valid_per) * B), :, :]
        else:  # is_train == 2, test set
            H = H[int((train_per + valid_per) * B):, :, :]
            cl = cl[int((train_per + valid_per) * B):, :, :]

        B, L, Nt = H.shape
        H_real = np.zeros([B, L, Nt, 2])
        H_real[:, :, :, 0] = H.real
        H_real[:, :, :, 1] = H.imag
        H_real = torch.tensor(H_real, dtype=torch.float32)
        cl = torch.tensor(cl, dtype=torch.float32)

        self.H = H_real
        self.cl = cl

    def __len__(self):
        return self.H.shape[0]

    def __getitem__(self, index):
        return {
            "H": self.H[index, :, :, :].float(),
            "cl": self.cl[index, :, :].float(),
        }


if __name__ == "__main__":
    """最小化数据加载测试"""
    import sys
    data_root = os.path.join(os.path.dirname(__file__), "Data_user.mat")

    print("=" * 50)
    print("数据接口验证测试")
    print("=" * 50)

    # 1. 检查文件是否存在
    if not os.path.exists(data_root):
        print(f"[错误] 找不到 {data_root}")
        sys.exit(1)
    print(f"[OK] 数据文件: {data_root}")

    # 2. 检查原始数据维度
    raw = sio.loadmat(data_root)
    H_raw = raw['h_near_slant']
    cl_raw = raw['index_far_near']
    print(f"\n[原始数据]")
    print(f"  h_near_slant shape: {H_raw.shape}, dtype: {H_raw.dtype}")
    print(f"  index_far_near shape: {cl_raw.shape}, dtype: {cl_raw.dtype}")

    # 3. 测试三个数据集
    for split_name, is_train in [("训练集", 1), ("验证集", 0), ("测试集", 2)]:
        dataset = ChannelDataset(data_root, is_train=is_train)
        H, cl = dataset.H, dataset.cl

        print(f"\n[{split_name}] (is_train={is_train})")
        print(f"  数据集长度: {len(dataset)}")
        print(f"  H shape: {H.shape}")
        print(f"  cl shape: {cl.shape}")

        # 检查 NaN/Inf
        has_nan = torch.isnan(H).any().item()
        has_inf = torch.isinf(H).any().item()
        print(f"  H NaN: {has_nan}, Inf: {has_inf}")

        # 检查 cl 分布
        cl_flat = cl.view(-1).numpy()
        n_zero = (cl_flat == 0).sum()
        n_one = (cl_flat == 1).sum()
        total = len(cl_flat)
        print(f"  cl 分布: 0(远场)={n_zero} ({100*n_zero/total:.1f}%), 1(近场)={n_one} ({100*n_one/total:.1f}%)")

        # 检查单个样本
        sample = dataset[0]
        print(f"  单样本 H shape: {sample['H'].shape}")
        print(f"  单样本 cl shape: {sample['cl'].shape}")

    print("\n" + "=" * 50)
    print("测试完成")
    print("=" * 50)
