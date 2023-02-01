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


num_reps = 5
int_epoch = 500
num_comp = np.arange(1,11)
data = torch.zeros((29,240,200))
sub=0
for file in list(glob.glob('../data/processed/*.h5')):
    data_subject = h5py.File(file, mode='r')
    data_tmp = torch.tensor(np.array(data_subject['data']))
    data[sub] = torch.transpose(data_tmp,dim0=0,dim1=1).float()
    sub+=1

for K in num_comp:
    synth_dataset = '../data/synthetic_methods/HMMdata_orig.h5'
    dataf = h5py.File(synth_dataset, mode='r')
    data = torch.tensor(np.array(dataf['X']))
    data = torch.unsqueeze(torch.transpose(data,dim0=0,dim1=1),dim=0).float()

    for m in range(4):
        for r in range(num_reps):
            thres_like = 10000000
            for r2 in range(num_reps):
                if m==0:
                    model = TorchMixtureModel(distribution_object=ACG,K=K, dist_dim=3)
                elif m==1:
                    model = HMM(num_states=K, observation_dim=3, emission_dist=ACG)
                elif m==2:
                    model = TorchMixtureModel(distribution_object=Watson,K=K, dist_dim=3)
                elif m==3:
                    model = HMM(num_states=K, observation_dim=3, emission_dist=Watson)

                optimizer = optim.Adam(model.parameters(), lr=0.1)
                if m==0 or m==2:
                    like = train_hmm(model, data=torch.squeeze(data), optimizer=optimizer, num_epoch=int_epoch, keep_bar=False,early_stopping=True)
                elif m==1 or m==3:
                    like = train_hmm(model, data=data, optimizer=optimizer, num_epoch=int_epoch, keep_bar=False,early_stopping=True)
                
                # load best model and calculate posterior or viterbi
                model.load_state_dict(torch.load('../data/interim/model_checkpoint.pt'))
                like_best = np.loadtxt('../data/interim/likelihood.txt')
                if like_best[1]<thres_like:
                    thres_like = like_best[1]
                    if m==0:
                        post = model.posterior(torch.squeeze(data))
                        np.savetxt('../data/synthetic_K/K'+str(K)+'ACG_MM_likelihood'+str(r)+'.csv',like_best)
                        np.savetxt('../data/synthetic_K/K'+str(K)+'ACG_MM_assignment'+str(r)+'.csv',np.transpose(post.detach()))
                    elif m==1:
                        best_path,xx,xxx = model.viterbi2(data)
                        np.savetxt('../data/synthetic_K/K'+str(K)+'ACG_HMM_likelihood'+str(r)+'.csv',like_best)
                        np.savetxt('../data/synthetic_K/K'+str(K)+'ACG_HMM_assignment'+str(r)+'.csv',np.transpose(best_path))
                    elif m==2:
                        post = model.posterior(torch.squeeze(data))
                        np.savetxt('../data/synthetic_K/K'+str(K)+'Watson_MM_likelihood'+str(r)+'.csv',like_best)
                        np.savetxt('../data/synthetic_K/K'+str(K)+'Watson_MM_assignment'+str(r)+'.csv',np.transpose(post.detach()))
                    elif m==3:
                        best_path,xx,xxx = model.viterbi2(data)
                        np.savetxt('../data/synthetic_K/K'+str(K)+'Watson_HMM_likelihood'+str(r)+'.csv',like_best)
                        np.savetxt('../data/synthetic_K/K'+str(K)+'Watson_HMM_assignment'+str(r)+'.csv',np.transpose(best_path))
