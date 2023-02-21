#!/usr/bin/env python
# coding: utf-8

# In[1]:


## Imports and get_param

import os
import sys
import glob

import h5py
import torch
from torch import optim
import numpy as np
from tqdm.notebook import trange, tqdm
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join('..')))

from src.models.AngularCentralGauss_torch import AngularCentralGaussian as ACG
from src.models.Watson_torch import Watson
from src.models.MixtureModel_torch import TorchMixtureModel
from src.models.HMM_torch import HiddenMarkovModel as HMM
from src.various.training import train_hmm

def get_param(model, show=True):
    para = model.get_model_param()
    
    if show:
        for p_k in para:
            print(p_k)
            print(para[p_k])
            print(10*'---')
    
    return para


num_repsouter = 5
num_repsinner = 10
LRs = np.array((0.001,0.01,0.1,1,10,100))
int_epoch = 100000
data = torch.zeros((29,240,100))
sub=0

datah5 = h5py.File('../data/processed/dataset_all_subjects_LEiDA_100.hdf5', 'r')
print(len(datah5.keys()))
for idx,subject in enumerate(list(datah5.keys())):
    data[idx] = torch.tensor(datah5[subject])
        
data_concat = torch.concatenate([data[sub] for sub in range(data.shape[0])])

def run_experiment(LR):
    for m in range(4):
        if m==0:
            model = TorchMixtureModel(distribution_object=ACG,K=3, dist_dim=data.shape[2],regu=1e-5)
            optimizer = optim.Adam(model.parameters(), lr=LR)
            like = train_hmm(model, data=data_concat, optimizer=optimizer, num_epoch=int_epoch, keep_bar=False,early_stopping=False)
        elif m==1:
            continue
            model = HMM(num_states=3, observation_dim=data.shape[2], emission_dist=ACG)
            optimizer = optim.Adam(model.parameters(), lr=LR)
            like = train_hmm(model, data=data, optimizer=optimizer, num_epoch=int_epoch, keep_bar=False,early_stopping=False)
        elif m==2:
            model = TorchMixtureModel(distribution_object=Watson,K=3, dist_dim=data.shape[2],regu=1e-5)
            optimizer = optim.Adam(model.parameters(), lr=LR)
            like = train_hmm(model, data=data_concat, optimizer=optimizer, num_epoch=int_epoch, keep_bar=False,early_stopping=False)
        elif m==3:
            continue
            model = HMM(num_states=3, observation_dim=data.shape[2], emission_dist=Watson)
            optimizer = optim.Adam(model.parameters(), lr=LR)
            like = train_hmm(model, data=data, optimizer=optimizer, num_epoch=int_epoch, keep_bar=False,early_stopping=False)


        if m==0:
            np.savetxt('../data/realLR/LR_'+np.array2string(LR)+'_ACG_MM_likelihood.csv',like)
        elif m==1:
            np.savetxt('../data/realLR/LR_'+np.array2string(LR)+'_ACG_HMM_likelihood.csv',like)
        elif m==2:
            np.savetxt('../data/realLR/LR_'+np.array2string(LR)+'_Watson_MM_likelihood.csv',like)
        elif m==3:
            np.savetxt('../data/realLR/LR_'+np.array2string(LR)+'_Watson_HMM_likelihood.csv',like)
if __name__=="__main__":
    run_experiment(LR=float(sys.argv[1]))
    #run_experiment(m=0)