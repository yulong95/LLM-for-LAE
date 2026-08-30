import numpy as np
import h5py
from torch.utils.data import Dataset
from einops import rearrange
import torch
import re

class MIMIDetectionDataset(Dataset):
    def __init__(self, data_root,is_train=True):

        self.data_root = data_root

        H = h5py.File(data_root,'r')['H'][:]  
        H = rearrange(H, 'n L a -> a L n')
        
        B,Nr,Nt = H.shape
        H_real = np.zeros([B,Nr,Nt, 2])
        H_real[:, :, :, 0] = H['real']
        H_real[:, :, :, 1] = H['imag']
        H_real = torch.tensor(H_real, dtype=torch.float)

        if is_train:
            self.H = H_real[:50000, :, :, :]
        else:
            self.H = H_real[:10000, :, :, :]
        
        self.cons = np.linspace(int(-np.sqrt(16)+1), int(np.sqrt(16)-1), int(np.sqrt(16)))
        self.alpha = np.sqrt((self.cons ** 2).mean())
        self.cons /= (self.alpha * np.sqrt(2))
        self.cons = torch.tensor(self.cons).to(dtype=torch.float32)

    
    def modulate(self, indices):
        x = self.cons[indices]
        return x

    def __len__(self):
        return self.H.shape[0]

    def __getitem__(self, index):
        
        instruction = '[MIMO Detection] The dataset is collected in a MIMO system with 4 transmit and 128 receive antennas with 16-QAM modulation. <Instruction> Recover the transmitted data given the channel information and received data attached.'
        
        instruction = text_processor(instruction)

        channel = self.H[index,:,:,:]
        Hr = channel[:,:,0]
        Hi = channel[:,:,1]        
        h1 = torch.cat((Hr, -1. * Hi), dim=1)
        h2 = torch.cat((Hi, Hr), dim=1)
        H_real = torch.cat((h1, h2), dim=0)

        indices_QAM = torch.randint(low=0, high=(np.int64(np.sqrt(16))), size=(2*8,6))
        ind_temp = rearrange(indices_QAM, '(W H) L -> W L H',H=2)
        real_part = ind_temp[:,:,0]
        complex_part = ind_temp[:,:,1]
        joint_indices = np.int64(np.sqrt(16))*real_part + complex_part
        x_det = self.modulate(indices_QAM)

        y = H_real @ x_det  # 或 torch.mm(H, x)
        signal_power = torch.mean(y ** 2)
        SNR_dB = torch.rand(1) * 20 +5.0 
        SNR_linear = 10 ** (SNR_dB / 10)  
        noise_power = signal_power / SNR_linear
        noise = torch.randn_like(y) * torch.sqrt(noise_power)
        y_noisy = y + noise

        return {
            "channel": self.H[index,:,:,:],
            "received_data": y_noisy.float(),
            "instruction_input": instruction,
            "ori_data": x_det.float(),
            "joint_indices": joint_indices,
            "noise_sigma": torch.sqrt(noise_power),
        }
    
def text_processor(caption):
        
    caption = re.sub(r"([.!\"()*#:;~])"," ",caption.lower(),)
    caption = re.sub(r"\s{2,}"," ",caption,)
    caption = caption.rstrip("\n")
    caption = caption.strip(" ")

    return caption

def noise(y, SNR,power_rx):
    log_noise = SNR-10*np.log10(power_rx)
    sigma = 10 ** (-log_noise/10)
    add_noise = np.sqrt(sigma / 2) * (np.random.randn(*y.shape) + 1j * np.random.randn(*y.shape))
    y = y['real'] + 1j * y['imag']
    return y + add_noise