import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
#from models.gpt2_model import Gpt2Model
from models.gpt2_model_quan import Gpt2Model
from datasets import build_dataset,build_dataloader
from utils import sel_criterion_train,sel_criterion_test,NMSELoss
from einops import rearrange
from torch.optim.lr_scheduler import CosineAnnealingLR

# ============= HYPER PARAMS(Pre-Defined) ==========#
lr = 0.0001
epochs = 500
batch_size = 50
device = torch.device('cuda:0')

class Params:
    def __init__(self):
        #self.model_path = "/gpt2-ori"
        self.model_path = "/gpt2-ori-4bit-16rank"
        self.predict_dimension_in = 48 
        self.predict_dimension_out = 48
        self.prev_len = 16
        self.pred_len = 4
        self.muh_dimen = 128
        self.sigma = 10**(-10/10)
params = Params()
model = Gpt2Model(params)

device = torch.device('cuda:0')
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)  
model.to(device)

data_root = {}
data_root['chanpre'] = "/multi-task-LLM/data_CP/train/CP_train.mat"
data_root['mimodet'] = "/multi-task-LLM/data_DET/train/DEC_train.mat"
data_root['mupre'] = "/multi-task-LLM/data_PRE/train/PRE_train.mat"
data_root_val= {}
data_root_val['chanpre'] = "/multi-task-LLM/data_CP/test/CP_val.mat"
data_root_val['mimodet'] = "/multi-task-LLM/data_DET/test/DEC_val.mat"
data_root_val['mupre'] = "/multi-task-LLM/data_PRE/test/PRE_val.mat"

output_dir = "/multi-task-LLM/output/"

w_cp = 1
w_det = 5
w_pre = 0.2

###################################################################
# ------------------- Main Train (Run second)----------------------
###################################################################
def train(trainloader_group,valloader_group):
    global epochs, best_loss
    best_loss = 100
    print('Start training...')
    for epoch in range(epochs):
        # ============Epoch Train=============== #
        since = time.time()
        epoch_train_loss_chanpre = []
        epoch_train_loss_mimodet = []
        epoch_train_loss_mupre = []
        epoch_train_loss = []

        model.train()
        data_tuple = [data_loader for data_loader in trainloader_group.values()]
        for iteration, data_batch in enumerate(zip(*data_tuple)):
            optimizer.zero_grad() 
            loss = 0
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
                elif ta.startswith('mupre'):
                    muchannel = data['muchannel'].to(device, non_blocking=True)
                    instruction2 = data['instruction_input']
                    p_wmmse = data['p_wmmse'].to(device, non_blocking=True)
                    lamda_wmmse = data['lamda_wmmse'].to(device, non_blocking=True)
                else:
                    raise NotImplementedError()
                
                # gpt2_model and loss computation
                if ta.startswith('chanpre'):
                    outputs = model(prev=prev,instruction=instruction0,ta_perform=ta)
                    loss_cp = criterion_train[ta](outputs*10**5, pred*10**5) 
                    loss = loss + loss_cp* w_cp
                elif ta.startswith('mimodet'):
                    outputs = model(channel=channel, received_data=y,instruction=instruction1,ta_perform=ta) 
                    x = rearrange(x, 'n (W H) L -> n L W H',H=2)
                    loss_det = criterion_train[ta](outputs*np.sqrt(10), x*np.sqrt(10))
                    loss = loss + loss_det* w_det 
                elif ta.startswith('mupre'):
                    random_int = np.random.randint(6)+3 #random 4-8 users
                    muchannel = muchannel[:,:random_int,:]
                    p_hat,lamda_hat = model(muchannel=muchannel, instruction=instruction2,ta_perform=ta)
                    loss_pre = -1.0 * criterion_train[ta](p_hat,lamda_hat,muchannel,params.sigma,params.muh_dimen)
                    loss = loss + loss_pre* w_pre
                
            loss.backward()
            optimizer.step()

            for ta in ta_sel:
                if ta.startswith('chanpre'):
                    epoch_train_loss_chanpre.append(loss_cp.item()) 
                elif ta.startswith('mimodet'):
                    epoch_train_loss_mimodet.append(loss_det.item()) 
                elif ta.startswith('mupre'):
                    epoch_train_loss_mupre.append(loss_pre.item())
            epoch_train_loss.append(loss.item())

            if (iteration+1) % 200==0:
                chanpre_loss = np.nanmean(np.array(epoch_train_loss_chanpre))
                mimodet_loss = np.nanmean(np.array(epoch_train_loss_mimodet))
                mupre_loss = np.nanmean(np.array(epoch_train_loss_mupre))
                print('Iteration: {} channel prediction loss: {:.7f} MIMO detection loss: {:.7f} multi-user precoding loss: {:.7f}'.format(iteration+1, chanpre_loss,mimodet_loss,mupre_loss))


        chanpre_loss = np.nanmean(np.array(epoch_train_loss_chanpre))
        mimodet_loss = np.nanmean(np.array(epoch_train_loss_mimodet))
        mupre_loss = np.nanmean(np.array(epoch_train_loss_mupre))
        loss_all = np.nanmean(np.array(epoch_train_loss))
        time_elapsed = time.time() - since
        print('Epoch: {}/{} channel prediction loss: {:.7f} MIMO detection loss: {:.7f} multi-user precoding loss: {:.7f} Training complete in {:.0f}m {:.0f}s'.format(epoch+1, epochs, chanpre_loss,mimodet_loss,mupre_loss,time_elapsed // 60, time_elapsed % 60))  # print loss for each epoch
        print('Overall loss: {:.7f}'.format(loss_all))
        scheduler.step() 

        if (epoch+1) % 5==0:
            model.eval()  
            val_cp_stack = []
            val_det_mse_stack = []
            val_det_error_stack = []
            val_pre_stack = []
            val_pre_wmmse_stack = []
            with torch.no_grad():
                data_tuple = [data_loader for data_loader in valloader_group.values()]
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
                            loss_cp = criterion_val[ta](outputs, pred)
                            val_cp_stack.append(loss_cp.item())
                        elif ta.startswith('mimodet'):
                            outputs = model(channel=channel, received_data=y,instruction=instruction1,ta_perform=ta) 
                            x = rearrange(x, 'n (W H) L -> n L W H',H=2)
                            loss_det = criterion1(outputs*np.sqrt(10), x*np.sqrt(10))
                            val_det_mse_stack.append(loss_det.item())
                            outputs0 = rearrange(outputs, 'n A W H -> n (A W) H')
                            j_indices = rearrange(j_indices, 'n A W -> n (W A)')
                            num_error = criterion_val[ta](outputs0*np.sqrt(10),j_indices)
                            val_det_error_stack.append(num_error.item())
                        elif ta.startswith('mupre'):
                            p_hat,lamda_hat = model(muchannel=muchannel, instruction=instruction2,ta_perform=ta)
                            rate = criterion_val[ta](p_hat,lamda_hat,muchannel,params.sigma,params.muh_dimen)
                            rate_stand = criterion_val[ta](p_wmmse,lamda_wmmse,muchannel,params.sigma,params.muh_dimen)
                            val_pre_stack.append(rate.item())
                            val_pre_wmmse_stack.append(rate_stand.item())
            
            total_loss = 0
            for ta in ta_sel:
                if ta.startswith('chanpre'):
                    nmse_cp = np.nanmean(np.array(val_cp_stack))
                    nmse_db_cp = 10*np.log10(nmse_cp)
                    print("channel prediction: NMSE:{:.7f},{:.7f}".format(nmse_cp,nmse_db_cp))
                    total_loss = total_loss + nmse_cp
                elif ta.startswith('mimodet'):
                    mse_dec = np.nanmean(np.array(val_det_mse_stack))
                    error_all = np.sum(np.array(val_det_error_stack))
                    ser = error_all/8.0/10000/8
                    print("MIMO detection: SER:{:.7f},NMSE:{:.7f}".format(ser,mse_dec))
                    total_loss = total_loss + mse_dec
                elif ta.startswith('mupre'):
                    rate = np.nanmean(np.array(val_pre_stack))
                    rate_wmmse = np.nanmean(np.array(val_pre_wmmse_stack))
                    print("Multi-user precoding: Rate:{:.7f}".format(rate))
                    print("Multi-user precoding: Rate_wmmse:{:.7f}".format(rate_wmmse))
                    total_loss = total_loss - rate

            if total_loss<best_loss:
                best_loss = total_loss
                torch.save(model.state_dict(), output_dir+'%s' % (epoch+1)+'.bin')


###################################################################
# ----------------------- Main Function ---------------------------
###################################################################

if __name__ == "__main__":

    print("------------------------------------------------------")
    ############## Get the data and dataloader ####################
    ta_sel = ['chanpre','mimodet','mupre']

    trainset_group = build_dataset(data_root=data_root,is_train=True, ta_sel=ta_sel)
    trainloader_group= build_dataloader(ta_sel,trainset_group,batch_size)

    valset_group = build_dataset(data_root=data_root_val,is_train=False, ta_sel=ta_sel)
    valloader_group= build_dataloader(ta_sel,valset_group,batch_size)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion_train = sel_criterion_train(ta_sel, device)
    criterion_val = sel_criterion_test(ta_sel, device)
    criterion1 = NMSELoss().to(device)

    total = sum([param.numel() for param in model.parameters()])
    print("Number of parameter: %.5fM" % (total / 1e6))
    total_learn = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of learnable parameter: %.5fM" % (total_learn / 1e6))

    train(trainloader_group,valloader_group)
    

