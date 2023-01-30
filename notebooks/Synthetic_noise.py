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
from src.various.training import train_hmm

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

noise_levels = np.linspace(-20,-80,9)
num_reps = 5
int_epoch = 500

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
            if m==0:
                model = TorchMixtureModel(distribution_object=ACG,K=2, dist_dim=3)
            elif m==1:
                model = HMM(num_states=2, observation_dim=3, emission_dist=ACG)
            elif m==2:
                model = TorchMixtureModel(distribution_object=Watson,K=2, dist_dim=3)
            elif m==3:
                model = HMM(num_states=2, observation_dim=3, emission_dist=Watson)

            optimizer = optim.Adam(model.parameters(), lr=0.1)
            if m==0 or m==2:
                like = train_hmm(model, data=torch.squeeze(data), optimizer=optimizer, num_epoch=int_epoch, keep_bar=False)
            elif m==1 or m==3:
                like = train_hmm(model, data=data, optimizer=optimizer, num_epoch=int_epoch, keep_bar=False)
            
            # load best model and calculate posterior or viterbi
            model.load_state_dict(torch.load('../data/interim/model_checkpoint.pt'))
            model.eval()
            like_best = np.loadtxt('../data/interim/likelihood.txt')
            if m==0:
                post = model.posterior(torch.squeeze(data))
                np.savetxt('../data/synthetic_noise/noise_'+str(noise)+'ACG_MM_likelihood'+str(r)+'.csv',like_best)
                np.savetxt('../data/synthetic_noise/noise_'+str(noise)+'ACG_MM_assignment'+str(r)+'.csv',np.transpose(post.detach()))
            elif m==1:
                best_path,xx,xxx = model.viterbi2(data)
                np.savetxt('../data/synthetic_noise/noise_'+str(noise)+'ACG_HMM_likelihood'+str(r)+'.csv',like_best)
                np.savetxt('../data/synthetic_noise/noise_'+str(noise)+'ACG_HMM_assignment'+str(r)+'.csv',np.transpose(best_path))
            elif m==2:
                post = model.posterior(torch.squeeze(data))
                np.savetxt('../data/synthetic_noise/noise_'+str(noise)+'Watson_MM_likelihood'+str(r)+'.csv',like_best)
                np.savetxt('../data/synthetic_noise/noise_'+str(noise)+'Watson_MM_assignment'+str(r)+'.csv',np.transpose(post.detach()))
            elif m==3:
                best_path = model.viterbi2(data)
                np.savetxt('../data/synthetic_noise/noise_'+str(noise)+'Watson_HMM_likelihood'+str(r)+'.csv',like_best)
                np.savetxt('../data/synthetic_noise/noise_'+str(noise)+'Watson_HMM_assignment'+str(r)+'.csv',np.transpose(best_path))
