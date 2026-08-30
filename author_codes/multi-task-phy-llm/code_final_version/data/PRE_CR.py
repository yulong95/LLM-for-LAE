import numpy as np
import h5py
from torch.utils.data import Dataset
from einops import rearrange
import torch
import re

class MultiuserPrecodingDataset(Dataset):
    def __init__(self, data_root,is_train=True):

        self.data_root = data_root

        H = h5py.File(data_root,'r')['H'][:]  
        H = rearrange(H, 'n L a -> a L n')
        p_wmmse = h5py.File(data_root,'r')["P"][:]  
        p_wmmse = rearrange(p_wmmse, 'n a -> a n')
        lamda_wmmse = h5py.File(data_root,'r')["Lamda"][:]  
        lamda_wmmse = rearrange(lamda_wmmse, 'n a -> a n')
        Pre = h5py.File(data_root,'r')['Pre'][:]  
        Pre = rearrange(Pre, 'n L a -> a L n') 

        H = rearrange(H, '(a b) L n -> a (L b) n',b=2)
        p_wmmse = rearrange(p_wmmse, '(a b) L -> a (L b)',b=2)
        lamda_wmmse = rearrange(lamda_wmmse, '(a b) L -> a (L b)',b=2)
        Pre = rearrange(Pre, '(a b) L n -> a (L b) n',b=2) 

        B,K,Nt = H.shape
        H_real = np.zeros([B,K,Nt, 2])
        H_real[:, :, :, 0] = H['real']
        H_real[:, :, :, 1] = H['imag']
        H_real = rearrange(H_real, 'b K N a -> b K (N a)')
        H_real = torch.tensor(H_real, dtype=torch.float)
        Pre_real = np.zeros([B,K,Nt, 2])
        Pre_real[:, :, :, 0] = Pre['real']
        Pre_real[:, :, :, 1] = Pre['imag']
        Pre_real = rearrange(Pre_real, 'b K N a -> b K (N a)')
        Pre_real = torch.tensor(Pre_real, dtype=torch.float)
        p_wmmse = torch.tensor(p_wmmse, dtype=torch.float)
        lamda_wmmse = torch.tensor(lamda_wmmse, dtype=torch.float)


        if is_train:
            self.H = H_real[:50000,:,:] 
            self.p_wmmse = p_wmmse[:50000,:] 
            self.lamda_wmmse = lamda_wmmse[:50000,:]
            self.Pre = Pre_real[:50000,:,:] 
        else:
            self.H = H_real[:10000,:,:]
            self.p_wmmse = p_wmmse[:10000,:] 
            self.lamda_wmmse = lamda_wmmse[:10000,:]
            self.Pre = Pre_real[:10000,:,:] 


    def __len__(self):
        return self.H.shape[0]

    def __getitem__(self, index):
        instruction = '[Multi-user precoding] For the collected dataset, we consider a base station with 128 antennas to serve 4-8 single antenna user simutaneously. <Instruction> Design the precoding matrix given channels of users, to maximize the sum rate of the multiple users.'
        
        instruction = text_processor(instruction)

        return {
            "muchannel": self.H[index,:,:],
            "instruction_input": instruction,
            "p_wmmse": self.p_wmmse[index,:],
            "lamda_wmmse": self.lamda_wmmse[index,:],
            "Pre": self.Pre[index,:,:]
        }

def text_processor(caption):
        
    caption = re.sub(r"([.!\"()*#:;~])"," ",caption.lower(),)
    caption = re.sub(r"\s{2,}"," ",caption,)
    caption = caption.rstrip("\n")
    caption = caption.strip(" ")

    return caption