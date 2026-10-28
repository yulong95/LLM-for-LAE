"""
A-mix 版 GPT2 / Transformer / CNN：输出每用户 A-row + p + cl，恢复 W=normalize(H A)。
与 mini_exp E4 同一套恢复；用于拉开 DL 间隙。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from models.transformer_model import TransformerEncoder
from models.amix_recovery import amix_to_V, hrate_from_H0, split_A_p_cl, c3_project

N, K = 256, 10
AMIX_OUT = 2 * K + 2  # A_row re/im + p + cl


class TF_Amix(nn.Module):
    def __init__(self, gamma=0.4, depth=5, heads=8, dim_head=64, mlp_dim=1536):
        super().__init__()
        self.gamma = gamma
        self.dim = N * 2
        self.input_norm = nn.LayerNorm(self.dim)
        self.encoder = TransformerEncoder(self.dim, depth=depth, heads=heads, dim_head=dim_head, mlp_dim=mlp_dim)
        self.head = nn.Linear(self.dim, AMIX_OUT)

    def forward(self, H0, cl, mean=None, std=None, use_pred_cl_for_c3=False):
        """H0: [B,K,N,2] or [B,K,2N] interleaved. cl: [B,K,1] or [B,K]."""
        if H0.dim() == 3:
            H0 = H0.reshape(H0.shape[0], H0.shape[1], N, 2)
        cls = cl.squeeze(-1)
        if mean is None:
            mean = torch.mean(H0)
        if std is None:
            std = torch.std(H0)
        H_rate = hrate_from_H0(H0)
        x = (H_rate - mean) / std
        x = self.encoder(self.input_norm(x))
        out = self.head(x)  # [B,K,2K+2]
        A, p, clh = split_A_p_cl(out, K)
        eye = torch.eye(K, device=H0.device, dtype=A.dtype).unsqueeze(0)
        A = eye + A
        cl_c3 = ((clh >= 0.5).float() if use_pred_cl_for_c3 else cls)
        p = c3_project(p, cl_c3, gamma=self.gamma)
        V = amix_to_V(H0, A, p)
        return V, H_rate, clh, p, A


class CNN_Amix(nn.Module):
    def __init__(self, gamma=0.4):
        super().__init__()
        self.gamma = gamma
        self.conv1 = nn.Conv2d(2, 12, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(12, momentum=0.99, eps=0.001)
        self.conv2 = nn.Conv2d(12, 12, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(12, momentum=0.99, eps=0.001)
        self.fc = nn.Linear(12 * N * K, 2 * K * K + 2 * K)

    def forward(self, H0, cl, mean=None, std=None, use_pred_cl_for_c3=False):
        if H0.dim() == 3:
            H0 = H0.reshape(H0.shape[0], H0.shape[1], N, 2)
        cls = cl.squeeze(-1)
        if mean is None:
            mean = torch.mean(H0)
        if std is None:
            std = torch.std(H0)
        H_rate = hrate_from_H0(H0)
        x = rearrange((H0 - mean) / std, 'b k n a -> b a n k')
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.view(-1, 12 * N * K)
        out = self.fc(x)
        Ar = out[:, :K * K].reshape(-1, K, K)
        Ai = out[:, K * K:2 * K * K].reshape(-1, K, K)
        A = torch.complex(Ar, Ai)
        eye = torch.eye(K, device=H0.device, dtype=A.dtype).unsqueeze(0)
        A = eye + A
        p = torch.sigmoid(out[:, 2 * K * K:2 * K * K + K])
        clh = torch.sigmoid(out[:, 2 * K * K + K:2 * K * K + 2 * K])
        cl_c3 = ((clh >= 0.5).float() if use_pred_cl_for_c3 else cls)
        p = c3_project(p, cl_c3, gamma=self.gamma)
        V = amix_to_V(H0, A, p)
        return V, H_rate, clh, p, A


class GPT2_Amix(nn.Module):
    """GPT-2 backbone + A-mix head (per-user 2K+2)."""

    def __init__(self, model_path, gamma=0.4, Nt=256, K=10, mlp=1):
        super().__init__()
        from models.gpt2_model_all import Gpt2Model
        # reuse init_llm / transformer_mu pieces via composition
        self.gamma = gamma
        self.Nt = Nt
        self.K = K
        self.hidden_size = 768
        self.mlp = mlp
        self.gpt2_model, self.gpt2_tokenizer = self._init_llm(model_path)
        from models.gpt2_model_all import Transformer
        self.transformer_mu = Transformer(self.Nt * 2, depth=3, heads=8, dim_head=64, mlp_dim=512, dropout=0.)
        self.llama_proj = nn.Linear(self.Nt * 2, self.hidden_size)
        self.head = nn.Linear(self.hidden_size, AMIX_OUT)

    def _init_llm(self, gpt2_model_path):
        import logging
        from transformers.models.gpt2.modeling_gpt2 import GPT2Model
        from transformers import GPT2Tokenizer
        logging.info('Loading GPT2 for A-mix')
        tok = GPT2Tokenizer.from_pretrained(gpt2_model_path, use_fast=False)
        tok.pad_token_id = 0
        gpt2 = GPT2Model.from_pretrained(gpt2_model_path, torch_dtype=torch.float32)
        for name, param in gpt2.named_parameters():
            if 'ln' in name or 'wpe' in name:
                param.requires_grad = True
            elif 'mlp' in name and self.mlp == 1:
                param.requires_grad = True
            else:
                param.requires_grad = False
        return gpt2, tok

    def forward(self, H0, cl, mean=None, std=None, use_pred_cl_for_c3=False):
        if H0.dim() == 3:
            H0 = H0.reshape(H0.shape[0], H0.shape[1], N, 2)
        cls = cl.squeeze(-1)
        if mean is None:
            mean = torch.mean(H0)
        if std is None:
            std = torch.std(H0)
        H_rate = hrate_from_H0(H0)
        x = (H_rate - mean) / std
        muchannel = self.transformer_mu(x)
        input_embs = self.llama_proj(muchannel)
        input_atts = torch.ones(input_embs.size()[:-1], dtype=torch.long).to(muchannel.device)
        last_hidden_state = self.gpt2_model(
            input_ids=None, inputs_embeds=input_embs, attention_mask=input_atts
        ).last_hidden_state
        out = self.head(last_hidden_state)  # [B,K,2K+2]
        A, p, clh = split_A_p_cl(out, K)
        eye = torch.eye(K, device=H0.device, dtype=A.dtype).unsqueeze(0)
        A = eye + A
        cl_c3 = ((clh >= 0.5).float() if use_pred_cl_for_c3 else cls)
        p = c3_project(p, cl_c3, gamma=self.gamma)
        V = amix_to_V(H0, A, p)
        return V, H_rate, clh, p, A
