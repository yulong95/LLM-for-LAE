import numpy as np
from torch.utils.data import Dataset
import h5py
import re
from torchvision import transforms
from einops import rearrange
import torch

class ChannelPredictionDataset(Dataset):
    def __init__(self, data_root,is_train=True):
        self.data_root = data_root
        self.data_transform = transforms.ToTensor()

        H_his = h5py.File(data_root,'r')['H_his_train'][:]   
        H_pre = h5py.File(data_root,'r')['H_pre_train'][:]  
        H_his = rearrange(H_his, 'n L k a -> (n a) k L')
        H_pre = rearrange(H_pre, 'n L k a -> (n a) k L')
        
        B, prev_len, mul = H_his.shape
        _, pred_len, mul = H_pre.shape
        self.pred_len = pred_len
        self.prev_len = prev_len
        self.seq_len = pred_len + prev_len

        H_real = np.zeros([B, prev_len, mul, 2])
        H_real[:, :, :, 0] = H_his['real']
        H_real[:, :, :, 1] = H_his['imag']
        H_real = H_real.reshape([B, prev_len, mul * 2])
        H_his = torch.tensor(H_real, dtype=torch.float)
        H_real = np.zeros([B, pred_len, mul, 2])
        H_real[:, :, :, 0] = H_pre['real']
        H_real[:, :, :, 1] = H_pre['imag']
        H_real = H_real.reshape([B, pred_len, mul * 2])
        H_pre = torch.tensor(H_real, dtype=torch.float)

        if is_train:
            self.pred = H_pre[:50000,:,:]
            self.prev = H_his[:50000,:,:]
        else: 
            self.pred = H_pre[:10000,:,:]
            self.prev = H_his[:10000,:,:]


    def __len__(self):
        return self.prev.shape[0]

    def __getitem__(self, index):
        
        instruction = '[Channel Prediction] The dataset is collected in an OFDM system, describing the time-varing channel of moving users over time. <Instruction> Predict the CSI for next 4 time steps given the previous 16 steps information attached.'
            
        instruction = text_processor(instruction)

        return {
            "prev": self.prev[index,:,:],
            "instruction_input": instruction,
            "pred": self.pred[index,:,:],
        }

def text_processor(caption):
        
    caption = re.sub(r"([.!\"()*#:;~])"," ",caption.lower(),)
    caption = re.sub(r"\s{2,}"," ",caption,)
    caption = caption.rstrip("\n")
    caption = caption.strip(" ")

    return caption
