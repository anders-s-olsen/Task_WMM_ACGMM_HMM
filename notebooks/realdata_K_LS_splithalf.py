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
num_repsinner = 1
int_epoch = 25000
num_comp = np.arange(1,11)
regus = np.flip(np.array((1e-08,1e-07,1e-06,1e-05,1e-04,1e-03,0.00215,0.00464,1e-02,0.0215,0.04641e-01,1e-00)))
num_regions = 100

sub=0

datah5 = h5py.File('../data/processed/dataset_all_subjects_LEiDA_100.hdf5', 'r')
#print(len(datah5.keys()))
data_train = torch.zeros((29,120,num_regions))
data_test = torch.zeros((29,120,num_regions))
for idx,subject in enumerate(list(datah5.keys())):
    data_train[idx] = torch.tensor(datah5[subject][0:120])
    data_test[idx] = torch.tensor(datah5[subject][120:])
        
data_train_concat = torch.concatenate([data_train[sub] for sub in range(data_train.shape[0])])
data_test_concat = torch.concatenate([data_test[sub] for sub in range(data_test.shape[0])])
#for m in range(4):
def run_experiment(K):
    for regu in regus:
        regustr = str(regu).replace('.','')
        if os.path.isfile('../data/real_K_LR/K'+str(K)+'regu'+regustr+'.csv'):
            continue
        print(regu)

        model = TorchMixtureModel(distribution_object=ACG,K=K, dist_dim=data_train.shape[2],regu=regu)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        like,model,like_best = train_hmm(model, data=data_train_concat, optimizer=optimizer, num_epoch=int_epoch, keep_bar=False,early_stopping=True)

        test_like = -model.log_likelihood_mixture(data_test_concat)

        

        np.savetxt('../data/real_K_LR/K'+str(K)+'regu'+regustr+'.csv',np.array((test_like.detach(),K,regu)))

if __name__=="__main__":
    run_experiment(K=int(sys.argv[1]))