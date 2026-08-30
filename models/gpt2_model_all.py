import os
import numpy as np
import logging
import torch
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers import GPT2Tokenizer
from einops import rearrange

class Gpt2Model(nn.Module):
    def __init__(self,
        model_path="", 
        Nt = 256,
        K=10,
        gamma=0.4,
        mlp = 1,
    ):
        super().__init__()

        self.hidden_size = 768
        self.mlp = mlp
        self.Nt = Nt
        self.K = K 
        self.P_max = 1 
        self.gamma = gamma   
        self.gpt2_model, self.gpt2_tokenizer = self.init_llm(gpt2_model_path=model_path)
        #encoder
        self.transformer_mu = Transformer(self.Nt*2, depth=3, heads=8, dim_head=64, mlp_dim=512, dropout=0.)
        self.llama_proj = nn.Linear(self.Nt*2, self.hidden_size)
        #decoder
        self.precoding_out_layer = nn.Linear(self.hidden_size, 3)
        self.output_layer = nn.Sigmoid()

    def init_llm(self,gpt2_model_path):
        logging.info('Loading GPT2')
        gpt2_tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_path, use_fast=False)
        gpt2_tokenizer.pad_token_id = 0
        gpt2 =  GPT2Model.from_pretrained(gpt2_model_path,torch_dtype=torch.float32)
        
        for i, (name, param) in enumerate(gpt2.named_parameters()):
            if 'ln' in name or 'wpe' in name:  
                param.requires_grad = True
            elif 'mlp' in name and self.mlp == 1:
                param.requires_grad = True 
            else:
                param.requires_grad = False

        logging.info('Loading GPT2 Done')
        return gpt2, gpt2_tokenizer
    
    def noise(self,H, SNR):
        sigma = 10 ** (- SNR / 10).to(H.device)
        add_noise = torch.sqrt(sigma / 2) * (torch.randn(*H.shape)).to(H.device)
        add_noise = add_noise * torch.sqrt(torch.mean(torch.abs(H) ** 2))
        return H + add_noise
    
    def forward(self, H=None,cl=None):
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

        last_hidden_state = self.gpt2_model(input_ids=None, inputs_embeds=input_embs,attention_mask=input_atts).last_hidden_state 

        dec_out = self.precoding_out_layer(last_hidden_state)
        dec_out = self.output_layer(dec_out)
        cl = cl.squeeze(-1)
        
        p_hat = dec_out[:,:,0]
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


        lamda_hat = dec_out[:,:,1]
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

        cl_hat = dec_out[:,:,2]
        return p_hat_0,lamda_hat_0,cl_hat

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)
    