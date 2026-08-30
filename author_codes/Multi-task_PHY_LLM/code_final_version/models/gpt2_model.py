import logging
import torch
import torch
import torch.nn as nn
from transformers import GPT2Model
from transformers import GPT2Tokenizer
from einops import rearrange
from einops.layers.torch import Rearrange
from peft import LoraConfig, get_peft_model

class Gpt2Model(nn.Module):
    def __init__(self,params):
        super().__init__()

        self.hidden_size = 768
        self.gpt2_model, self.gpt2_tokenizer = self.init_llm(gpt2_model_path=params.model_path)

        # channel prediction encoder
        self.patch_size = 4
        self.res_dim  = 50
        self.res_layers = 4
        self.patch_layer = nn.Linear(self.patch_size, self.patch_size)
        self.RB_e = nn.Sequential(nn.Conv2d(2, self.res_dim, 3, 1, 1))
        for i in range(self.res_layers):
            self.RB_e.append(Res_block(self.res_dim))
        self.RB_e.append(nn.Conv2d(self.res_dim, 2, 3, 1, 1))
        self.gpt2_proj1 = nn.Linear(params.predict_dimension_in*2, self.hidden_size)
        self.predict_linear_pre = nn.Linear(params.prev_len, params.prev_len)

        # channel prediction decoder
        self.predict_out_layer = nn.Linear(self.hidden_size, params.predict_dimension_out * 2)
        self.output_layer_time = nn.Linear(params.prev_len, params.pred_len)

        # detection encoder
        self.Nr = 128
        self.Nt = 8
        self.patch_size_dec = 4
        self.embed_dim_dec = self.Nr*self.Nt*2
        self.det_proj1 = nn.Linear(self.Nr*2,self.Nr*2)
        self.det_proj2 = nn.Linear(self.Nr*2, self.hidden_size)
        self.det_proj3 = nn.Linear(self.hidden_size, self.hidden_size)
        patch_dim = 2 * self.patch_size_dec * self.patch_size_dec
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = self.patch_size_dec, p2 = self.patch_size_dec),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, self.embed_dim_dec),
            nn.LayerNorm(self.embed_dim_dec),
        )
        num_patches = (self.Nr // self.patch_size_dec) * (self.Nt // self.patch_size_dec)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, self.embed_dim_dec))
        self.emb_dropout = 0.
        self.dropout = nn.Dropout(self.emb_dropout)
        self.transformer = Transformer(self.embed_dim_dec, depth=3, heads=8, dim_head=64, mlp_dim=64, dropout=0.)
        self.gpt2_proj2 = nn.Linear(self.embed_dim_dec, self.hidden_size)

        # detection decoder
        self.detection_out_layer = nn.Linear(self.hidden_size, self.Nt*2)
        self.output_data = nn.Linear(self.Nt*2, self.Nt*2)

        # precoding encoder
        self.transformer_mu = Transformer(params.muh_dimen*2, depth=3, heads=8, dim_head=64, mlp_dim=256, dropout=0.)
        self.gpt2_proj3 = nn.Linear(params.muh_dimen*2, self.hidden_size)

        # precoding decoder
        self.precoding_out_layer = nn.Linear(self.hidden_size, 2)
        self.output_layer = nn.Sigmoid()

    def init_llm(self,gpt2_model_path):
        logging.info('Loading GPT2')
        gpt2_tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_path, use_fast=False)
        gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
        gpt2 =  GPT2Model.from_pretrained(gpt2_model_path,torch_dtype=torch.float32)

        logging.info('Loading GPT2 Done')

        target_modules=["c_attn", "c_proj", "c_fc"] 
        lora_config = LoraConfig(
            r=16,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=target_modules,
        )
        gpt2 = get_peft_model(gpt2, lora_config)

        logging.info('Setting LORA')

        for i, (name, param) in enumerate(gpt2.named_parameters()):
            if 'ln' in name or 'wpe' in name:  
                param.requires_grad = True

        total_trainable = 0
        for name, param in gpt2.named_parameters():
            if param.requires_grad:
                total_trainable += param.numel()
        print(f"\nGPT2总可训练参数数量: {total_trainable:,}")

        return gpt2, gpt2_tokenizer
       
    def predict_encoder(self,prev,instruction):
        mean = torch.mean(prev)
        std = torch.std(prev)
        prev = (prev - mean) / std

        B, L, enc_in = prev.shape
        for i in range(B):
            SNR = torch.rand(1) * 20
            #SNR = torch.tensor(10.0)
            prev[i, ...] = self.noise(prev[i, ...], SNR)
        x_enc = prev.reshape(B, L // self.patch_size, self.patch_size, enc_in)
        x_enc = self.patch_layer(x_enc.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        x_enc = x_enc.reshape(B, L, enc_in)
        x_enc = rearrange(x_enc, 'b l (k o) -> b o l k', o=2)
        x_enc = self.RB_e(x_enc)
        x_enc = rearrange(x_enc, 'b o l k -> b l (k o)')  # [B, L, D]
        x_enc = self.gpt2_proj1(x_enc)
        prev_embeds = self.predict_linear_pre(x_enc.permute(0, 2, 1)).permute(0, 2, 1)
        atts_prev = torch.ones(prev_embeds.size()[:-1], dtype=torch.long).to(prev.device)

        if instruction!=None:
            self.gpt2_tokenizer.padding_side = "right"
            prompt_tokens = self.gpt2_tokenizer(
                instruction,
                return_tensors="pt",
                padding="longest",
                add_special_tokens=False
            )
            prompt_embeds = self.embed_tokens(prompt_tokens.input_ids.to(prev.device))
            atts_prompt = prompt_tokens.attention_mask.to(prev.device)

            input_embs = torch.cat((prompt_embeds, prev_embeds), dim=1)
            input_atts = torch.cat((atts_prompt, atts_prev), dim=1)
            input_lens = torch.sum(input_atts,dim=1)
        else:
            input_embs = prev_embeds
            input_atts = atts_prev
            input_lens = torch.sum(atts_prev,dim=1)

        return input_embs,input_atts,mean,std,input_lens

    def detection_encoder(self,channel,received_data,instruction):
        data_embeds = self.det_proj1(received_data)
        data_embeds = self.det_proj2(data_embeds)
        data_embeds = self.det_proj3(data_embeds)
        atts_data = torch.ones(data_embeds.size()[:-1], dtype=torch.long).to(channel.device)

        mean = torch.mean(channel)
        std = torch.std(channel)
        channel = (channel - mean) / std
        channel = rearrange(channel, 'B Nr Nt a -> B a Nt Nr')
        channel = self.to_patch_embedding(channel)
        channel += self.pos_embedding
        channel = self.dropout(channel)
        channel = self.transformer(channel)
        channel_embeds = self.gpt2_proj2(channel)
        atts_channel = torch.ones(channel_embeds.size()[:-1], dtype=torch.long).to(channel.device)

        if instruction!=None:
            self.gpt2_tokenizer.padding_side = "right"
            prompt_tokens = self.gpt2_tokenizer(
                instruction,
                return_tensors="pt",
                padding="longest",
                add_special_tokens=False
            )
            prompt_embeds = self.embed_tokens(prompt_tokens.input_ids.to(channel.device))
            atts_prompt = prompt_tokens.attention_mask.to(channel.device)

            input_embs = torch.cat((prompt_embeds, channel_embeds, data_embeds), dim=1)
            input_atts = torch.cat((atts_prompt, atts_channel, atts_data), dim=1)
            input_lens = torch.sum(input_atts,dim=1)
        else:
            input_embs = torch.cat((channel_embeds, data_embeds), dim=1)
            input_atts = torch.cat((atts_channel, atts_data), dim=1)
            input_lens = torch.sum(input_atts,dim=1)

        return input_embs,input_atts,input_lens

    def precoding_encoder(self,muchannel,instruction):
        mean = torch.mean(muchannel)
        std = torch.std(muchannel)
        muchannel = (muchannel - mean) / std
        B, L, enc_in = muchannel.shape
        for i in range(B):
            SNR = torch.rand(1) * 15 + 5.0
            #SNR = torch.tensor(10.0)
            muchannel[i, ...] = self.noise(muchannel[i, ...], SNR)
        muchannel = self.transformer_mu(muchannel)
        muchannel_embeds = self.gpt2_proj3(muchannel)
        atts_muchannel = torch.ones(muchannel_embeds.size()[:-1], dtype=torch.long).to(muchannel.device)

        if instruction!=None:
            self.gpt2_tokenizer.padding_side = "right"
            prompt_tokens = self.gpt2_tokenizer(
                instruction,
                return_tensors="pt",
                padding="longest",
                add_special_tokens=False
            )
            prompt_embeds = self.embed_tokens(prompt_tokens.input_ids.to(muchannel.device))
            atts_prompt = prompt_tokens.attention_mask.to(muchannel.device)

            input_embs = torch.cat((prompt_embeds, muchannel_embeds), dim=1)
            input_atts = torch.cat((atts_prompt, atts_muchannel), dim=1)
            input_lens = torch.sum(input_atts,dim=1)
        else:
            input_embs = muchannel_embeds
            input_atts = atts_muchannel
            input_lens = torch.sum(atts_muchannel,dim=1)

        return input_embs,input_atts,input_lens

    def predict_decoder(self,last_hidden_state,mean,std):
        pred_out = self.predict_out_layer(last_hidden_state)
        pred_out = pred_out[:,-16:,:]
        dec_out = self.output_layer_time(pred_out.permute(0, 2, 1)).permute(0, 2, 1)
        pred_hat = dec_out * std + mean
        return pred_hat

    def detection_decoder(self,last_hidden_state):
        dec_out = self.detection_out_layer(last_hidden_state)
        dec_out = dec_out[:,-6:,:]
        data_hat = self.output_data(dec_out)
        B,L,Nt = data_hat.shape
        data_hat = torch.reshape(data_hat,[B,L,Nt//2,2])
        return data_hat

    def precoding_decoder(self,last_hidden_state,L):
        dec_out = last_hidden_state[:,-L:,:]
        dec_out = self.precoding_out_layer(dec_out)
        dec_out = self.output_layer(dec_out)
        p_hat = dec_out[:,:,0]
        p_sum = torch.norm(p_hat,p=2,dim=1, keepdim=True)**2
        p_hat = p_hat/torch.sqrt(p_sum+ 1e-8) 
        lamda_hat = dec_out[:,:,1]
        lamda_sum = torch.sum(lamda_hat, dim=1, keepdim=True)
        lamda_hat = lamda_hat/ (lamda_sum+ 1e-8) 
        return p_hat,lamda_hat

    def embed_tokens(self, token_ids):
        embeds = self.gpt2_model.wte(token_ids)
        return embeds

    def noise(self,H, SNR):
        sigma = 10 ** (- SNR / 10).to(H.device)
        add_noise = torch.sqrt(sigma / 2) * (torch.randn(*H.shape)).to(H.device)
        L, enc_in = H.shape
        add_noise = add_noise * torch.sqrt(torch.mean(torch.abs(H) ** 2)/L)
        return H + add_noise

    def forward(self, prev=None, channel=None, received_data=None, muchannel=None, ta_perform=None, instruction=None):
        if ta_perform.startswith('chanpre'):
            input_embs,input_atts,mean,std,input_lens = self.predict_encoder(prev,instruction)
        elif ta_perform.startswith('mimodet'):
            input_embs,input_atts,input_lens = self.detection_encoder(channel,received_data,instruction)
        elif ta_perform.startswith('mupre'):
            b,L,d = muchannel.shape
            input_embs,input_atts,input_lens = self.precoding_encoder(muchannel,instruction)

        last_hidden_state = self.gpt2_model(input_ids=None, inputs_embeds=input_embs,attention_mask=input_atts).last_hidden_state 

        if ta_perform.startswith('chanpre'):
            pred_hat = self.predict_decoder(last_hidden_state,mean,std)
            return pred_hat
        elif ta_perform.startswith('mimodet'):
            data_hat = self.detection_decoder(last_hidden_state)
            return data_hat
        elif ta_perform.startswith('mupre'):
            p_hat,lamda_hat = self.precoding_decoder(last_hidden_state,L)
            return p_hat,lamda_hat

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=4):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class Res_block(nn.Module):
    def __init__(self, in_planes):
        super(Res_block, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, in_planes, 3, 1, 1)
        self.conv2 = nn.Conv2d(in_planes, in_planes, 3, 1, 1)
        self.ca = ChannelAttention(in_planes=in_planes, ratio=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        rs1 = self.relu(self.conv1(x))
        rs1 = self.conv2(rs1)
        channel_attn = self.ca(rs1)
        output = channel_attn * rs1
        rs = torch.add(x, output)
        return rs

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
