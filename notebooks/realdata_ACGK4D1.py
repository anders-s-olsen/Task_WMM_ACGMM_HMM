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
from src.models.AngularCentralGauss_lowrank import AngularCentralGaussian as ACG_lowrank
from src.models.Watson_torch import Watson
from src.models.MixtureModel_torch import TorchMixtureModel
from src.models.HMM_torch import HiddenMarkovModel as HMM
from src.various.training import train_hmm_batch, train_hmm,train_hmm_lbfgs,train_hmm_subject

torch.set_num_threads(16)

def get_param(model, show=False):
    para = model.get_model_param()
    
    if show:
        for p_k in para:
            print(p_k)
            print(para[p_k])
            print(10*'---')
    
    return para


def run_experiment(K):
    num_regions = 100

    datah5 = h5py.File('../data/processed/dataset_all_subjects_LEiDA_100.hdf5', 'r')
    #print(len(datah5.keys()))
    data_train = torch.zeros((29,120,num_regions),dtype=torch.double)
    data_test = torch.zeros((29,120,num_regions),dtype=torch.double)
    for idx,subject in enumerate(list(datah5.keys())):
        data_train[idx] = torch.DoubleTensor(np.array(datah5[subject][0:120]))
        data_test[idx] = torch.DoubleTensor(np.array(datah5[subject][120:]))

    #for m in range(4):
    model0 = TorchMixtureModel(distribution_object=ACG_lowrank,K=K, D=1,dist_dim=data_train.shape[2])
    optimizer = optim.Adam(model0.parameters(), lr=0.1)
    like_ACG = train_hmm_batch(model0, data=data_train, optimizer=optimizer, num_epoch=1000, keep_bar=False,early_stopping=False,modeltype=0)
    
    model1 = TorchMixtureModel(distribution_object=Watson,K=K, dist_dim=data_train.shape[2])
    optimizer = optim.Adam(model1.parameters(), lr=0.1)
    like_Watson = train_hmm_batch(model1, data=data_train.to(torch.float32), optimizer=optimizer, num_epoch=5000, keep_bar=False,early_stopping=False,modeltype=0)
    param = get_param(model1)
    init = {}
    init['pi'] = param['un_norm_pi']
    init['comp'] = torch.zeros((K,data_train.shape[2],1),dtype=torch.double)
    for kk in range(K):
        init['comp'][kk] = torch.sqrt(param['mix_comp_'+str(kk)]['kappa'])*torch.unsqueeze(param['mix_comp_'+str(kk)]['mu'],dim=1)
    
    model2 = TorchMixtureModel(distribution_object=ACG_lowrank,K=K, D=1,dist_dim=data_train.shape[2],init=init)
    optimizer = optim.Adam(model2.parameters(), lr=0.1)
    like_ACG_init = train_hmm_batch(model2, data=data_train, optimizer=optimizer, num_epoch=1000, keep_bar=False,early_stopping=False,modeltype=0)
    
    p0 = get_param(model0)
    p1 = get_param(model1)
    p2 = get_param(model2)
    plt.figure()
    fig,ax = plt.subplots(5,3)

    for k in range(K):
        ax[k+1,0].imshow((p0['mix_comp_'+str(k)]@p0['mix_comp_'+str(k)].T).detach())
    for k in range(K):
        ax[k+1,1].imshow(torch.sqrt(p1['mix_comp_'+str(k)]['kappa'])*torch.outer(p1['mix_comp_'+str(k)]['mu'],p1['mix_comp_'+str(k)]['mu']).detach())
        
    for k in range(K):
        ax[k+1,2].imshow((p2['mix_comp_'+str(k)]@p2['mix_comp_'+str(k)].T).detach())

    ax[0,0].plot(like_ACG)
    ax[0,1].plot(like_Watson)
    ax[0,2].plot(like_ACG_init)
    fig.show()
    y=8
                
if __name__=="__main__":
    run_experiment(K=4)