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


num_repsouter = 3
num_repsinner = 5
int_epoch = 500
num_comp = np.arange(1,11)
data = torch.zeros((29,240,200))
sub=0

datah5 = h5py.File('../data/processed/dataset_all_subjects_LEiDA.hdf5', 'r')
print(len(datah5.keys()))
for idx,subject in enumerate(list(datah5.keys())):
    data[idx] = torch.tensor(datah5[subject])
        
data_concat = torch.concatenate([data[sub] for sub in range(data.shape[0])])

for K in num_comp:
    for m in range(4):
        for r in range(num_repsouter):
            thres_like = 10000000
            for r2 in range(num_repsinner):
                if m==0:
                    model = TorchMixtureModel(distribution_object=ACG,K=K, dist_dim=data.shape[2])
                elif m==1:
                    model = HMM(num_states=K, observation_dim=data.shape[2], emission_dist=ACG)
                elif m==2:
                    model = TorchMixtureModel(distribution_object=Watson,K=K, dist_dim=data.shape[2])
                elif m==3:
                    model = HMM(num_states=K, observation_dim=data.shape[2], emission_dist=Watson)

                optimizer = optim.Adam(model.parameters(), lr=0.1)
                if m==0 or m==2:
                    like = train_hmm(model, data=data_concat, optimizer=optimizer, num_epoch=int_epoch, keep_bar=False,early_stopping=True)
                elif m==1 or m==3:
                    like = train_hmm(model, data=data, optimizer=optimizer, num_epoch=int_epoch, keep_bar=False,early_stopping=True)
                
                # load best model and calculate posterior or viterbi
                model.load_state_dict(torch.load('../data/interim/model_checkpoint.pt'))
                like_best = np.loadtxt('../data/interim/likelihood.txt')
                if like_best[1]<thres_like:
                    thres_like = like_best[1]
                    param = get_param(model)
                    if m==0:
                        post = model.posterior(data_concat)
                        np.savetxt('../data/real_K/K'+str(K)+'ACG_MM_likelihood'+str(r)+'.csv',like_best)
                        np.savetxt('../data/real_K/K'+str(K)+'ACG_MM_assignment'+str(r)+'.csv',np.transpose(post.detach()))
                        np.savetxt('../data/real_K/K'+str(K)+'ACG_MM_prior'+str(r)+'.csv',torch.nn.functional.softmax(param['un_norm_pi'],dim=0).detach())
                        for kk in range(K):
                            np.savetxt('../data/real_K/K'+str(K)+'ACG_MM_comp0'+str(r)+'.csv',param['mix_comp_'+str(kk)].detach())
                    elif m==1:
                        best_path,xx,xxx = model.viterbi2(data)
                        np.savetxt('../data/real_K/K'+str(K)+'ACG_HMM_likelihood'+str(r)+'.csv',like_best)
                        np.savetxt('../data/real_K/K'+str(K)+'ACG_HMM_assignment'+str(r)+'.csv',np.transpose(best_path))
                        np.savetxt('../data/real_K/K'+str(K)+'ACG_HMM_prior'+str(r)+'.csv',torch.nn.functional.softmax(param['un_norm_priors'],dim=0).detach())
                        np.savetxt('../data/real_K/K'+str(K)+'ACG_HMM_T'+str(r)+'.csv',torch.nn.functional.softmax(param['un_norm_Transition_matrix'],dim=1).detach())
                        for kk in range(K):
                            np.savetxt('../data/real_K/K'+str(K)+'ACG_HMM_comp0'+str(r)+'.csv',param['emission_model_'+str(kk)].detach())
                    elif m==2:
                        post = model.posterior(data_concat)
                        np.savetxt('../data/real_K/K'+str(K)+'Watson_MM_likelihood'+str(r)+'.csv',like_best)
                        np.savetxt('../data/real_K/K'+str(K)+'Watson_MM_assignment'+str(r)+'.csv',np.transpose(post.detach()))
                        np.savetxt('../data/real_K/K'+str(K)+'Watson_MM_prior'+str(r)+'.csv',torch.nn.functional.softmax(param['un_norm_pi'],dim=0).detach())
                        for kk in range(K):
                            np.savetxt('../data/real_K/K'+str(K)+'Watson_MM_comp0mu'+str(r)+'.csv',param['mix_comp_'+str(kk)]['mu'].detach())
                            np.savetxt('../data/real_K/K'+str(K)+'Watson_MM_comp0kappa'+str(r)+'.csv',param['mix_comp_'+str(kk)]['kappa'].detach())
                    elif m==3:
                        best_path,xx,xxx = model.viterbi2(data)
                        np.savetxt('../data/real_K/K'+str(K)+'Watson_HMM_likelihood'+str(r)+'.csv',like_best)
                        np.savetxt('../data/real_K/K'+str(K)+'Watson_HMM_assignment'+str(r)+'.csv',np.transpose(best_path))
                        np.savetxt('../data/real_K/K'+str(K)+'Watson_HMM_prior'+str(r)+'.csv',torch.nn.functional.softmax(param['un_norm_priors'],dim=0).detach())
                        np.savetxt('../data/real_K/K'+str(K)+'Watson_HMM_T'+str(r)+'.csv',torch.nn.functional.softmax(param['un_norm_Transition_matrix'],dim=1).detach())
                        for kk in range(K):
                            np.savetxt('../data/real_K/K'+str(K)+'Watson_HMM_comp0mu'+str(r)+'.csv',param['emission_model_'+str(kk)]['mu'].detach())
                            np.savetxt('../data/real_K/K'+str(K)+'Watson_HMM_comp0kappa'+str(r)+'.csv',param['emission_model_'+str(kk)]['kappa'].detach())
