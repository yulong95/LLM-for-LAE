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
