import torch
import torch.nn as nn
from einops import rearrange

def sel_criterion_train(ta_sel, device):
    criterion_group = {}
    for ta in ta_sel:
        if ta.startswith('chanpre'):
            criterion = torch.nn.MSELoss().to(device)
        elif ta.startswith('mimodet'):
            criterion = torch.nn.MSELoss().to(device)
        elif ta.startswith('mupre'):
            #criterion = torch.nn.MSELoss().to(device) #phase 1
            criterion = RateCal().to(device) #phase 2
        criterion_group[ta] = criterion
    return criterion_group

def sel_criterion_test(ta_sel, device):
    criterion_group = {}
    for ta in ta_sel:
        if ta.startswith('chanpre'):
            criterion = NMSELoss().to(device)
        elif ta.startswith('mimodet'):
            criterion = SERLoss().to(device)
            #criterion = NMSELoss().to(device)
        elif ta.startswith('mupre'):
            criterion = RateCal().to(device)
        criterion_group[ta] = criterion
    return criterion_group

# NMSE Loss
def NMSE_cuda(x_hat, x):
    power = torch.sum(x ** 2)
    mse = torch.sum((x - x_hat) ** 2)
    nmse = mse / power
    return nmse

class NMSELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(NMSELoss, self).__init__()
        self.reduction = reduction

    def forward(self, x_hat, x):
        nmse = NMSE_cuda(x_hat, x)
        if self.reduction == 'mean':
            nmse = torch.mean(nmse)
        else:
            nmse = torch.sum(nmse)
        return nmse

# SER Loss
def qamdemod(data_hat):
    import torch
    data_hat_real = data_hat[:, :, 0].unsqueeze(2)  # (B, 8, 1)
    data_hat_imag = data_hat[:, :, 1].unsqueeze(2)  # (B, 8, 1)
    constel_real = torch.tensor([-3, -3, -3, -3, -1, -1, -1, -1,1,1,1,1,3,3,3,3]).unsqueeze(0).unsqueeze(0).to(data_hat_real.device)  
    constel_imag = torch.tensor([-3, -1, 1, 3, -3, -1, 1, 3,-3,-1,1,3,-3,-1,1,3]).unsqueeze(0).unsqueeze(0).to(data_hat_real.device)  

    diff_real = data_hat_real - constel_real  # (B, 8, 16)
    diff_imag = data_hat_imag - constel_imag  # (B, 8, 16)
    dist = torch.sqrt(diff_real**2 + diff_imag**2)  # (B, 8, 16)
    symbol_hat = torch.argmin(dist, dim=2)  # (B, 8)

    return symbol_hat

class SERLoss(nn.Module):
    def __init__(self):
        super(SERLoss, self).__init__()
    
    def forward(self,data_hat,symbol):
        symbol_hat = qamdemod(data_hat)
        not_equal = symbol_hat != symbol
        num_diff = not_equal.sum()

        return num_diff

# Negative Rate Loss
class RateCal(nn.Module):
    def __init__(self):
        super(RateCal, self).__init__()

    def forward(self,p_hat,lamda_hat,muchannel,sigma,muh_dimen):
        precoding_mat = pq2V(p_hat,lamda_hat,muchannel,sigma,muh_dimen)
        Rsum = SMR_loss(precoding_mat,muchannel,sigma)
        Rsum = torch.mean(Rsum)
        return Rsum

def pq2V(p_hat,lamda_hat,muchannel,sigma,muh_dimen):
    Nt = muh_dimen
    K = p_hat.shape[1]
    sigma_2 = sigma
    muchannel = rearrange(muchannel, 'b K (N a c) -> b N c a K',a=2,c=1)
    channel = muchannel
    batch_size = channel.shape[0]
    p_list = p_hat
    q_list = lamda_hat
    channel = torch.view_as_complex(channel.permute(0,1,2,4,3).contiguous())
    P = channel[:,:,0,:]
    p_list = p_list.type_as(channel)
    q_list = q_list.type_as(channel)
    weighted_P = P * torch.sqrt(q_list).reshape((-1, 1, K)).repeat(1, Nt, 1)
    B = sigma_2 * torch.eye(K).type_as(channel).reshape((1, K, K)).repeat(batch_size, 1, 1)
    temp = weighted_P
    B = B + torch.matmul(torch.conj(temp.permute(0, 2, 1)), temp)
    P = torch.matmul(weighted_P, torch.linalg.inv(B))
    V = []
    for user in range(K):
        V_temp = P[:, :, user]/torch.sqrt(q_list[:, user]+1e-24).reshape((-1, 1)).repeat(1, Nt)
        V_temp = (p_list[:, user]).reshape((-1, 1)).repeat(1, Nt) * V_temp/(torch.norm(V_temp,dim=1).reshape(-1, 1).repeat(1, Nt) + 1e-48)
        V.append(V_temp)
    V = torch.conj(torch.stack(V, dim=2).reshape((-1, Nt, 1,K,1)))
    V = torch.cat((V.real,V.imag),dim=-1)
    return V

def SMR_loss(precoding_mat,muchannel,sigma):
        Nt = 128
        Nr = 1
        dk = 1
        K = muchannel.shape[1]
        sigma_2 = sigma
        muchannel = rearrange(muchannel, 'b K (N a c) -> b N c a K',a=2,c=1)
        y_true = muchannel

        y_pred = precoding_mat
        batch_size = y_true.shape[0]
        H = torch.view_as_complex(y_true.permute(0,1,2,4,3).contiguous())

        #restore V 
        V = torch.view_as_complex(y_pred.contiguous())
        '''precode matrix normalize'''              
        V_flatten = V.reshape((-1,Nt*dk*K))
        energy_scale = torch.linalg.norm(V_flatten,axis=1).reshape((-1,1,1,1)).repeat(1,Nt,dk,K).type_as(H)
        V = V/energy_scale
        '''need to change for normal runing'''
        sum_rate = torch.zeros(1).to('cuda:0')
        for user in range(K):
            H_k = H[:,:,:,user].permute(0,2,1)
            V_k = V[:,:,:,user]
            signal_k = torch.matmul(H_k, V_k)
            signal_k_energy = torch.matmul(signal_k,torch.conj(signal_k.permute(0,2,1)))
            interference_k_energy = sigma_2 * torch.eye(Nr).type_as(H).reshape((1,Nr,Nr)).repeat(batch_size,1,1)
            for j in range(K):
                if j!=user:
                    V_j = V[:, :, :, j]
                    interference_j = torch.matmul(H_k, V_j)
                    interference_k_energy = interference_k_energy + torch.matmul(interference_j,torch.conj(interference_j.permute(0,2,1)))
                SINR_k = torch.matmul(signal_k_energy, torch.linalg.inv(interference_k_energy))
                rate_k = torch.log2(complex_det(SINR_k + torch.eye(Nr).type_as(H).reshape((1,Nr,Nr)).repeat(batch_size,1,1)))
            sum_rate = sum_rate + rate_k
        return torch.mean(sum_rate)

def complex_det(A):
    A_real = A.real
    A_imag = A.imag
    upper_matrix = torch.cat((A_real,-A_imag),dim = 2)
    lower_matrix = torch.cat((A_imag,A_real),dim = 2)
    Matrix = torch.cat(((upper_matrix,lower_matrix)),dim = 1)
    det_result = torch.linalg.det(Matrix)
    return torch.sqrt(det_result)
