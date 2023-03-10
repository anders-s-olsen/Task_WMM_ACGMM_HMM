#!/usr/bin/env python
# coding: utf-8

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

from models.AngularCentralGauss_chol import AngularCentralGaussian as ACG
from src.models.AngularCentralGauss_lowrank import AngularCentralGaussian as ACG_lowrank
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
synth_dataset = '../data/synthetic_methods/HMMdata_orig.h5'
dataf = h5py.File(synth_dataset, mode='r')
data = torch.tensor(np.array(dataf['X']))
data = torch.unsqueeze(torch.transpose(data,dim0=0,dim1=1),dim=0).float()

num_reps = 2
best_LR = 0.1
int_epoch = 5000
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)

for m in tqdm(range(1)):
    best_like = 10000000
    for idx in tqdm(range(num_reps)):
        if m==0:
            model = TorchMixtureModel(distribution_object=ACG,K=2, dist_dim=3)
            #model = TorchMixtureModel(distribution_object=ACG_lowrank,K=2, dist_dim=3,D=2)
        elif m==1:
            model = HMM(num_states=2, observation_dim=3, emission_dist=ACG)
        elif m==2:
            model = TorchMixtureModel(distribution_object=Watson,K=2, dist_dim=3)
        elif m==3:
            model = HMM(num_states=2, observation_dim=3, emission_dist=Watson)

        optimizer = optim.Adam(model.parameters(), lr=best_LR)
        if m==0 or m==2:
            like,model,like_best = train_hmm(model, data=torch.squeeze(data), optimizer=optimizer, num_epoch=int_epoch, keep_bar=False,early_stopping=True)
            #like = train_hmm(model, data=torch.squeeze(data), optimizer=optimizer, num_epoch=int_epoch, keep_bar=False,early_stopping=False)
        elif m==1 or m==3:
            like,model,like_best = train_hmm(model, data=data, optimizer=optimizer, num_epoch=int_epoch, keep_bar=False,early_stopping=True)
        param = get_param(model)
        print(param['mix_comp_0']/torch.max(param['mix_comp_0']))
        print(param['mix_comp_1']/torch.max(param['mix_comp_1']))
        if like_best[-1] < best_like:
            best_model = model
            best_like = like_best[-1]
            best_idx = idx
    if m==0:
        best_ACG_MM = best_model
    elif m==1:
        best_ACG_HMM = best_model
    elif m==2:
        best_Watson_MM = best_model
    elif m==3:
        best_Watson_HMM = best_model

# ## extract parameters from the best model (should be pi around 0.5)

# In[14]:


acgmm_param = get_param(best_ACG_MM)
ACG_MM_post = best_ACG_MM.posterior(torch.squeeze(data))
np.savetxt('../data/synthetic_methods/ACG_MM_prior.csv',torch.nn.functional.softmax(acgmm_param['un_norm_pi'],dim=0).detach())
np.savetxt('../data/synthetic_methods/ACG_MM_comp0.csv',acgmm_param['mix_comp_0'].detach())
np.savetxt('../data/synthetic_methods/ACG_MM_comp1.csv',acgmm_param['mix_comp_1'].detach())
np.savetxt('../data/synthetic_methods/ACG_MM_posterior.csv',np.transpose(ACG_MM_post.detach()))

watsonmm_param = get_param(best_Watson_MM)
Watson_MM_post = best_Watson_MM.posterior(torch.squeeze(data))
np.savetxt('../data/synthetic_methods/Watson_MM_prior.csv',torch.nn.functional.softmax(watsonmm_param['un_norm_pi'],dim=0).detach())
np.savetxt('../data/synthetic_methods/Watson_MM_comp0mu.csv',watsonmm_param['mix_comp_0']['mu'].detach())
np.savetxt('../data/synthetic_methods/Watson_MM_comp0kappa.csv',watsonmm_param['mix_comp_0']['kappa'].detach())
np.savetxt('../data/synthetic_methods/Watson_MM_comp1mu.csv',watsonmm_param['mix_comp_1']['mu'].detach())
np.savetxt('../data/synthetic_methods/Watson_MM_comp1kappa.csv',watsonmm_param['mix_comp_1']['kappa'].detach())
np.savetxt('../data/synthetic_methods/Watson_MM_posterior.csv',np.transpose(Watson_MM_post.detach()))

acghmm_param = get_param(best_ACG_HMM)
ACG_HMM_best_paths, ACG_HMM_paths_probs, ACG_HMM_emission_probs = best_ACG_HMM.viterbi2(data)
np.savetxt('../data/synthetic_methods/ACG_HMM_prior.csv',torch.nn.functional.softmax(acghmm_param['un_norm_priors'],dim=0).detach())
np.savetxt('../data/synthetic_methods/ACG_HMM_T.csv',torch.nn.functional.softmax(acghmm_param['un_norm_Transition_matrix'],dim=1).detach())
np.savetxt('../data/synthetic_methods/ACG_HMM_comp0.csv',acghmm_param['emission_model_0'].detach())
np.savetxt('../data/synthetic_methods/ACG_HMM_comp1.csv',acghmm_param['emission_model_1'].detach())
np.savetxt('../data/synthetic_methods/ACG_HMM_viterbi.csv',np.transpose(ACG_HMM_best_paths))
np.savetxt('../data/synthetic_methods/ACG_HMM_emissionprobs.csv',np.squeeze(ACG_HMM_emission_probs))

watsonhmm_param = get_param(best_Watson_HMM)
Watson_HMM_best_paths, Watson_HMM_paths_probs, Watson_HMM_emission_probs = best_Watson_HMM.viterbi2(data)
np.savetxt('../data/synthetic_methods/Watson_HMM_prior.csv',torch.nn.functional.softmax(watsonhmm_param['un_norm_priors'],dim=0).detach())
np.savetxt('../data/synthetic_methods/Watson_HMM_T.csv',torch.nn.functional.softmax(watsonhmm_param['un_norm_Transition_matrix'],dim=1).detach())
np.savetxt('../data/synthetic_methods/Watson_HMM_comp0mu.csv',watsonhmm_param['emission_model_0']['mu'].detach())
np.savetxt('../data/synthetic_methods/Watson_HMM_comp0kappa.csv',watsonhmm_param['emission_model_0']['kappa'].detach())
np.savetxt('../data/synthetic_methods/Watson_HMM_comp1mu.csv',watsonhmm_param['emission_model_1']['mu'].detach())
np.savetxt('../data/synthetic_methods/Watson_HMM_comp1kappa.csv',watsonhmm_param['emission_model_1']['kappa'].detach())
np.savetxt('../data/synthetic_methods/Watson_HMM_viterbi.csv',np.transpose(Watson_HMM_best_paths))
np.savetxt('../data/synthetic_methods/Watson_HMM_emissionprobs.csv',np.squeeze(Watson_HMM_emission_probs))

