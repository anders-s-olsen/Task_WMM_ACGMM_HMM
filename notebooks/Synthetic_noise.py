#!/usr/bin/env python
# coding: utf-8

# In[1]:


## Imports and get_param

import os
import sys

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
from src.various.training import train_hmm,train_hmm_batch

def get_param(model, show=True):
    para = model.get_model_param()
    
    if show:
        for p_k in para:
            print(p_k)
            print(para[p_k])
            print(10*'---')
    
    return para


# ## Load data and get correct shape and dtype

# In[2]:

noise_levels = np.arange(-60,1,10)
num_reps = 1
int_epoch = 500
LR = 0.1
torch.set_num_threads(16)

for noise in noise_levels:
    try:
        synth_dataset = '../data/synthetic_noise/HMMdata_noise_'+np.array2string(noise)+'.h5'
        dataf = h5py.File(synth_dataset, mode='r')
    except:
        synth_dataset = '../data/synthetic_noise/HMMdata_noise_'+np.array2string(noise)+'h5'
        dataf = h5py.File(synth_dataset, mode='r')
    data = torch.tensor(np.array(dataf['X']))
    data = torch.unsqueeze(torch.transpose(data,dim0=0,dim1=1),dim=0).float()
    for m in range(4):
        for r in range(num_reps):
            thres_like = 10000000
            for r2 in range(num_reps):
                if m==0:
                    model = TorchMixtureModel(distribution_object=ACG,K=2, dist_dim=3)
                    optimizer = optim.Adam(model.parameters(), lr=LR)
                    like,model,like_best = train_hmm_batch(model, data, optimizer=optimizer, num_epoch=int_epoch, keep_bar=False,early_stopping=True,modeltype=0)
                elif m==1:
                    model = HMM(num_states=2, observation_dim=3, emission_dist=ACG)
                    optimizer = optim.Adam(model.parameters(), lr=LR)
                    like,model,like_best = train_hmm_batch(model, data=data, optimizer=optimizer, num_epoch=int_epoch, keep_bar=False,early_stopping=True,modeltype=1)
                elif m==2:
                    model = TorchMixtureModel(distribution_object=Watson,K=2, dist_dim=3)
                    optimizer = optim.Adam(model.parameters(), lr=LR)
                    like,model,like_best = train_hmm_batch(model, data, optimizer=optimizer, num_epoch=int_epoch, keep_bar=False,early_stopping=True,modeltype=0)
                elif m==3:
                    model = HMM(num_states=2, observation_dim=3, emission_dist=Watson)
                    optimizer = optim.Adam(model.parameters(), lr=LR)
                    like,model,like_best = train_hmm_batch(model, data=data, optimizer=optimizer, num_epoch=int_epoch, keep_bar=False,early_stopping=True,modeltype=1)


                if like_best[1]<thres_like:
                    thres_like = like_best[1]
                    if m==0:
                        post = model.posterior(torch.squeeze(data))
                        np.savetxt('../data/synthetic_noise/noise_'+str(noise)+'ACG_MM_likelihood'+str(r)+'.csv',like_best)
                        np.savetxt('../data/synthetic_noise/noise_'+str(noise)+'ACG_MM_assignment'+str(r)+'.csv',np.transpose(post.detach()))
                    elif m==1:
                        best_path,xx,xxx = model.viterbi2(data,external_eval=False)
                        np.savetxt('../data/synthetic_noise/noise_'+str(noise)+'ACG_HMM_likelihood'+str(r)+'.csv',like_best)
                        np.savetxt('../data/synthetic_noise/noise_'+str(noise)+'ACG_HMM_assignment'+str(r)+'.csv',np.transpose(best_path))
                    elif m==2:
                        post = model.posterior(torch.squeeze(data))
                        np.savetxt('../data/synthetic_noise/noise_'+str(noise)+'Watson_MM_likelihood'+str(r)+'.csv',like_best)
                        np.savetxt('../data/synthetic_noise/noise_'+str(noise)+'Watson_MM_assignment'+str(r)+'.csv',np.transpose(post.detach()))
                    elif m==3:
                        param = get_param(model)
                        best_path,xx,xxx = model.viterbi2(data,external_eval=False)
                        np.savetxt('../data/synthetic_noise/noise_'+str(noise)+'Watson_HMM_likelihood'+str(r)+'.csv',like_best)
                        np.savetxt('../data/synthetic_noise/noise_'+str(noise)+'Watson_HMM_assignment'+str(r)+'.csv',np.transpose(best_path))
