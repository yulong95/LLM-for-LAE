import torch
import numpy as np
import torch.nn as nn
from einops import rearrange
import torch
import torch.nn.functional as F


class ACCLoss(nn.Module):
    def __init__(self):
        super(ACCLoss, self).__init__()
    
    def forward(self,cl_targets,outputs,threshold = 0.5):
        cl_hat = (outputs >= threshold).float()
        correct_predictions = (cl_hat == cl_targets).float()
        accuracy = correct_predictions.sum() / cl_targets.shape[0] /cl_targets.shape[1]
        return accuracy

class RateCal(nn.Module):
    def __init__(self):
        super(RateCal, self).__init__()

    def forward(self,p_hat,lamda_hat,muchannel):
        sigma = 10**(-20/10)
        muh_dimen = 256
        K = p_hat.shape[1]
        precoding_mat = pq2V(p_hat,lamda_hat,muchannel,sigma,muh_dimen)
        Rsum,single_rate  = SMR_loss(precoding_mat,muchannel,sigma)
        Rsum = torch.mean(Rsum)
        return Rsum

class MULoss(nn.Module):
    def __init__(self):
        super(MULoss, self).__init__()

    def forward(self,p_hat,lamda_hat,muchannel):
        sigma = 10**(-20/10)
        muh_dimen = 256
        K = p_hat.shape[1]
        precoding_mat = pq2V(p_hat,lamda_hat,muchannel,sigma,muh_dimen)
        Rsum,single_rate = SMR_loss(precoding_mat,muchannel,sigma)
        penalty = F.relu(0.6 - single_rate)*10.0
        penalty = torch.sum(penalty,dim=1)
        penalty_loss = torch.mean(penalty)
        Rsum_loss = -1.0*torch.mean(Rsum)
        loss = Rsum_loss + penalty_loss
        return loss


def pq2V(p_hat,lamda_hat,muchannel,sigma,muh_dimen):
    Nt = muh_dimen
    Nr = 1
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
        Nt = 256
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
        # No global normalization — power allocation is handled by p[user] in pq2V
        # Each V_k has norm = p[user], so signal power = p[user] * |H_k*v_k|²
        sum_rate = torch.zeros(batch_size).cuda()
        single_rate = torch.zeros(batch_size,K).cuda()
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
            single_rate[:,user] = rate_k
            sum_rate = sum_rate + rate_k
        #sum_rate = - sum_rate
        return sum_rate,single_rate

def complex_det(A):
    A_real = A.real
    A_imag = A.imag
    upper_matrix = torch.cat((A_real,-A_imag),dim = 2)
    lower_matrix = torch.cat((A_imag,A_real),dim = 2)
    Matrix = torch.cat(((upper_matrix,lower_matrix)),dim = 1)
    det_result = torch.linalg.det(Matrix)
    return torch.sqrt(det_result)
