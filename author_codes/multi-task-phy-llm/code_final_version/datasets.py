import torch
from data.CP_CR import ChannelPredictionDataset
from data.DE_CR import MIMIDetectionDataset
from data.PRE_CR import MultiuserPrecodingDataset

def build_dataset(data_root, is_train, ta_sel):
    datasets = {}
    for ta in ta_sel:
        if  ta.startswith('chanpre'):
            dataset = ChannelPredictionDataset(data_root['chanpre'],is_train)
        elif  ta.startswith('mimodet'):
            dataset = MIMIDetectionDataset(data_root['mimodet'],is_train)
        elif ta.startswith('mupre'):
            dataset = MultiuserPrecodingDataset(data_root['mupre'],is_train)
        else:
            raise NotImplementedError()
        
        datasets[ta] = dataset

    return datasets

def build_dataloader(ta_sel, trainsets, batch_size):
    trainloaders = {}
    for ta in ta_sel:
        trainset = trainsets[ta]
        trainloader = torch.utils.data.DataLoader(dataset=trainset,
                                                pin_memory=True,batch_size=batch_size, shuffle=True)
        trainloaders[ta] = trainloader
    return trainloaders
