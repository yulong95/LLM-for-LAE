import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math
from einops import rearrange


class CNN_pre(nn.Module):
    def __init__(self, N, K,gamma):
        super(CNN_pre, self).__init__()
        self.N = N
        self.K = K
        self.gamma = gamma
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=12, kernel_size=3, padding=1, stride=1)
        self.bn1 = nn.BatchNorm2d(12, momentum=0.99, eps=0.001)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(12, momentum=0.99, eps=0.001)
        self.fc = nn.Linear(in_features=12*N*K, out_features=3*K)
        self.P_max = 1

    def noise(self,H, SNR):
        sigma = 10 ** (- SNR / 10).to(H.device)
        add_noise = torch.sqrt(sigma / 2) * (torch.randn(*H.shape)).to(H.device)
        add_noise = add_noise * torch.sqrt(torch.mean(torch.abs(H) ** 2))
        return H + add_noise

    def forward(self, x,cl,index,mean,std):
        x = (x - mean) / std
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.view(-1, 12*self.N*self.K)
        x = torch.sigmoid(self.fc(x))
        x = rearrange(x, 'n (H a) -> n H a',a=3)

        cl = cl.squeeze(-1)
        p_hat = x[:,:index,0]
        p_sum = torch.norm(p_hat,p=2,dim=1, keepdim=True)**2
        p_normalized = p_hat/torch.sqrt(p_sum+ 1e-8)
        temp_label_0 = torch.norm(p_normalized * cl,p=2,dim=1, keepdim=True)**2
        mask0 = temp_label_0 < self.gamma

        norm_label_1 = torch.norm(p_hat * cl,p=2,dim=1, keepdim=True)**2
        scale_1 = torch.sqrt(self.P_max * self.gamma /(norm_label_1 + 1e-8) )
        normalized_label_1 = p_hat * cl * scale_1
        norm_label_0 = torch.norm(p_hat * (1-cl),p=2,dim=1, keepdim=True)**2
        scale_0 = torch.sqrt(self.P_max * (1-self.gamma) /(norm_label_0 + 1e-8) )
        normalized_label_0 = p_hat * (1 - cl) * scale_0
        p_hat_0 = normalized_label_1+normalized_label_0
        p_hat_0 = mask0 * p_normalized  + (~mask0) * p_hat_0


        lamda_hat = x[:,:index,1]
        lamda_sum = torch.sum(lamda_hat, dim=1, keepdim=True)
        lamda_normalized = lamda_hat/ (lamda_sum+ 1e-8) 
        temp_label_1 = torch.sum(lamda_normalized * cl,dim=1, keepdim=True)
        mask1 = temp_label_1 < self.gamma

        sum_label_1 = torch.sum(lamda_hat * cl, dim=1, keepdim=True)
        scale_p_1 = self.P_max * self.gamma /(sum_label_1 + 1e-8) 
        normalized_P_label_1 = lamda_hat * cl * scale_p_1
        sum_label_0 = torch.sum(lamda_hat * (1 - cl), dim=1, keepdim=True)
        scale_p_0 = self.P_max * (1-self.gamma) /(sum_label_0 + 1e-8) 
        normalized_P_label_0 = lamda_hat * (1 - cl) * scale_p_0
        lamda_hat_0 = normalized_P_label_1+normalized_P_label_0
        lamda_hat_0 = mask1 * lamda_normalized + (~mask1) * lamda_hat_0

        cl_hat = x[:,:index,2]


        return p_hat_0,lamda_hat_0,cl_hat