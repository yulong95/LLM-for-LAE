import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from models.gpt2_model import Gpt2Model
#from models.gpt2_model_quan import Gpt2Model
from datasets import build_dataset,build_dataloader
from utils import sel_criterion_test,NMSELoss
from einops import rearrange
from torch.optim.lr_scheduler import CosineAnnealingLR

# ============= HYPER PARAMS(Pre-Defined) ==========#
class Params:
    def __init__(self):
        self.model_path = "/gpt2-ori"
        #self.model_path = "/gpt2-ori-4bit-16rank"
        self.predict_dimension_in = 48 
        self.predict_dimension_out = 48
        self.prev_len = 16
        self.pred_len = 4
        self.muh_dimen = 128
        self.sigma = 10**(-10/10)
params = Params()
model = Gpt2Model(params)

batch_size = 50
device = torch.device('cuda:0')
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)  
model.to(device)

data_root_test= {}
data_root_test['chanpre'] = "/multi-task-LLM/data_CP/test/CP_test_mean.mat"
data_root_test['mimodet'] = "/multi-task-LLM/data_DET/test/DEC_test_qua1.mat"
data_root_test['mupre'] = "/multi-task-LLM/data_PRE/test/PRE_test.mat"

output_dir = "/multi-task-LLM/output/"

model.load_state_dict(torch.load(output_dir+'300.bin'))

###################################################################
# ------------------- Main Train (Run second)----------------------
###################################################################

def evaluate(testloader_group):
    test_cp_stack = []
    test_det_stack = []
    test_pre_stack = []

    model.eval()
    with torch.no_grad():
        data_tuple = [data_loader for data_loader in testloader_group.values()]
        for iteration, data_batch in enumerate(zip(*data_tuple)):
            for task_idx, data in enumerate(data_batch):
                ta = ta_sel[task_idx]
                # data preparation
                if ta.startswith('chanpre'):
                    prev = data['prev'].to(device, non_blocking=True)
                    instruction0 = data['instruction_input']
                    pred = data['pred'].to(device, non_blocking=True)
                elif ta.startswith('mimodet'):
                    channel = data['channel'].to(device, non_blocking=True)
                    instruction1 = data['instruction_input']
                    y = data['received_data'].to(device, non_blocking=True)
                    y = rearrange(y, 'n W L -> n L W')
                    x = data['ori_data'].to(device, non_blocking=True)
                    j_indices = data['joint_indices'].to(device, non_blocking=True)
                elif ta.startswith('mupre'):
                    muchannel = data['muchannel'].to(device, non_blocking=True)
                    instruction2 = data['instruction_input']
                    p_wmmse = data['p_wmmse'].to(device, non_blocking=True)
                    lamda_wmmse = data['lamda_wmmse'].to(device, non_blocking=True)
                    muchannel = muchannel[:,:4,:]
                    p_wmmse = p_wmmse[:,:4]
                    lamda_wmmse = lamda_wmmse[:,:4]
                else:
                    raise NotImplementedError()
            
                # gpt2_model and loss computation
                if ta.startswith('chanpre'):
                    outputs = model(prev=prev,instruction=instruction0,ta_perform=ta)
                    loss_cp = criterion_test[ta](outputs, pred)
                    test_cp_stack.append(loss_cp.item())
                elif ta.startswith('mimodet'):
                    outputs = model(channel=channel, received_data=y,instruction=instruction1,ta_perform=ta) 
                    x = rearrange(x, 'n (W H) L -> n L W H',H=2)
                    # loss_det = criterion_test[ta](outputs*np.sqrt(10), x*np.sqrt(10))
                    # test_det_stack.append(loss_det.item())
                    outputs0 = rearrange(outputs, 'n A W H -> n (A W) H')
                    j_indices = rearrange(j_indices, 'n A W -> n (W A)')
                    num_error = criterion_test[ta](outputs0*np.sqrt(10),j_indices)
                    test_det_stack.append(num_error.item())
                elif ta.startswith('mupre'):
                    p_hat,lamda_hat = model(muchannel=muchannel, instruction=instruction2,ta_perform=ta)
                    rate = criterion_test[ta](p_hat,lamda_hat,muchannel,params.sigma,params.muh_dimen)
                    test_pre_stack.append(rate.item())
        
        for ta in ta_sel:
            if ta.startswith('chanpre'):
                nmse_cp = np.nanmean(np.array(test_cp_stack))
                nmse_db_cp = 10*np.log10(nmse_cp)
                print("channel prediction: NMSE:{:.7f},{:.7f}".format(nmse_cp,nmse_db_cp))
            elif ta.startswith('mimodet'):
                # mse_dec = np.nanmean(np.array(test_det_stack))
                # print("MIMO detection: NMSE:{:.7f}".format(mse_dec))
                error_all = np.sum(np.array(test_det_stack))
                ser = error_all/8.0/10000/8
                print("MIMO detection: SER:{:.7f}".format(ser))
            elif ta.startswith('mupre'):
                rate = np.nanmean(np.array(test_pre_stack))
                print("Multi-user precoding: Rate:{:.7f}".format(rate))


###################################################################
# ----------------------- Main Function ---------------------------
###################################################################

if __name__ == "__main__":

    print("------------------------------------------------------")
    ############## Get the data and dataloader ####################
    ta_sel = ['chanpre','mimodet','mupre']

    testset_group = build_dataset(data_root=data_root_test,is_train=False, ta_sel=ta_sel)
    testloader_group= build_dataloader(ta_sel,testset_group,batch_size)
    criterion_test = sel_criterion_test(ta_sel, device)

    evaluate(testloader_group)
