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


synth_dataset = '../data/synthetic/HMMdata.h5'
dataf = h5py.File(synth_dataset, mode='r')
data = torch.tensor(np.array(dataf['X']))
data = torch.unsqueeze(torch.transpose(data,dim0=0,dim1=1),dim=0).float()
#print(data.shape) #needs to be subjects, time, dims
#print(data.dtype)
#print(torch.norm(data,dim=2))


# ## ACG mixture
# train model with diff learning rates, get best model 

# In[7]:

int_epoch = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)

eval_LR = np.logspace(0.001,1,8)
for LR in eval_LR:
    for m in range(4):
        
        if m==0:
            model = TorchMixtureModel(distribution_object=ACG,K=2, dist_dim=3)
        elif m==1:
            #continue
            model = HMM(num_states=2, observation_dim=3, emission_dist=ACG)
        elif m==2:
            model = TorchMixtureModel(distribution_object=Watson,K=2, dist_dim=3)
        elif m==3:
            #continue
            model = HMM(num_states=2, observation_dim=3, emission_dist=Watson)

        optimizer = optim.Adam(model.parameters(), lr=LR)
        if m==0 or m==2:
            like = train_hmm(model, data=torch.squeeze(data), optimizer=optimizer, num_epoch=int_epoch, keep_bar=False)
        elif m==1 or m==3:
            like = train_hmm(model, data=data, optimizer=optimizer, num_epoch=int_epoch, keep_bar=False)
        
        if m==0:
            best_ACG_MM = model
            like_ACG_MM = like
        elif m==1:
            best_ACG_HMM = model
            like_ACG_HMM = like
        elif m==2:
            best_Watson_MM = model
            like_Watson_MM = like
        elif m==3:
            best_Watson_HMM = model
            like_Watson_HMM = like



    acgmm_param = get_param(best_ACG_MM)
    ACG_MM_post = best_ACG_MM.posterior(torch.squeeze(data))
    np.savetxt('../data/syntheticLR/LR_',np.array2string(LR),'ACG_MM_prior.csv',torch.nn.functional.softmax(acgmm_param['un_norm_pi'],dim=0).detach())
    np.savetxt('../data/syntheticLR/LR_',np.array2string(LR),'ACG_MM_comp0.csv',acgmm_param['mix_comp_0'].detach())
    np.savetxt('../data/syntheticLR/LR_',np.array2string(LR),'ACG_MM_comp1.csv',acgmm_param['mix_comp_1'].detach())
    np.savetxt('../data/syntheticLR/LR_',np.array2string(LR),'ACG_MM_posterior.csv',np.transpose(ACG_MM_post.detach()))
    np.savetxt('../data/syntheticLR/LR_',np.array2string(LR),'ACG_MM_likelihood.csv',like_ACG_MM.detach())

    watsonmm_param = get_param(best_Watson_MM)
    Watson_MM_post = best_Watson_MM.posterior(torch.squeeze(data))
    np.savetxt('../data/syntheticLR/LR_',np.array2string(LR),'Watson_MM_prior.csv',torch.nn.functional.softmax(watsonmm_param['un_norm_pi'],dim=0).detach())
    np.savetxt('../data/syntheticLR/LR_',np.array2string(LR),'Watson_MM_comp0mu.csv',watsonmm_param['mix_comp_0']['mu'].detach())
    np.savetxt('../data/syntheticLR/LR_',np.array2string(LR),'Watson_MM_comp0kappa.csv',watsonmm_param['mix_comp_0']['kappa'].detach())
    np.savetxt('../data/syntheticLR/LR_',np.array2string(LR),'Watson_MM_comp1mu.csv',watsonmm_param['mix_comp_1']['mu'].detach())
    np.savetxt('../data/syntheticLR/LR_',np.array2string(LR),'Watson_MM_comp1kappa.csv',watsonmm_param['mix_comp_1']['kappa'].detach())
    np.savetxt('../data/syntheticLR/LR_',np.array2string(LR),'Watson_MM_posterior.csv',np.transpose(Watson_MM_post.detach()))
    np.savetxt('../data/syntheticLR/LR_',np.array2string(LR),'Watson_MM_likelihood.csv',like_Watson_MM.detach())

    acghmm_param = get_param(best_ACG_HMM)
    ACG_HMM_best_paths, ACG_HMM_paths_probs, ACG_HMM_emission_probs = best_ACG_HMM.viterbi2(data)
    np.savetxt('../data/syntheticLR/LR_',np.array2string(LR),'ACG_HMM_prior.csv',torch.nn.functional.softmax(acghmm_param['un_norm_priors'],dim=0).detach())
    np.savetxt('../data/syntheticLR/LR_',np.array2string(LR),'ACG_HMM_T.csv',torch.nn.functional.softmax(acghmm_param['un_norm_Transition_matrix'],dim=1).detach())
    np.savetxt('../data/syntheticLR/LR_',np.array2string(LR),'ACG_HMM_comp0.csv',acghmm_param['emission_model_0'].detach())
    np.savetxt('../data/syntheticLR/LR_',np.array2string(LR),'ACG_HMM_comp1.csv',acghmm_param['emission_model_1'].detach())
    np.savetxt('../data/syntheticLR/LR_',np.array2string(LR),'ACG_HMM_viterbi.csv',np.transpose(ACG_HMM_best_paths))
    np.savetxt('../data/syntheticLR/LR_',np.array2string(LR),'ACG_HMM_emissionprobs.csv',np.squeeze(ACG_HMM_emission_probs))
    np.savetxt('../data/syntheticLR/LR_',np.array2string(LR),'ACG_HMM_likelihood.csv',like_ACG_HMM.detach())


    watsonhmm_param = get_param(best_Watson_HMM)
    Watson_HMM_best_paths, Watson_HMM_paths_probs, Watson_HMM_emission_probs = best_Watson_HMM.viterbi2(data)
    np.savetxt('../data/syntheticLR/LR_',np.array2string(LR),'Watson_HMM_prior.csv',torch.nn.functional.softmax(watsonhmm_param['un_norm_priors'],dim=0).detach())
    np.savetxt('../data/syntheticLR/LR_',np.array2string(LR),'Watson_HMM_T.csv',torch.nn.functional.softmax(watsonhmm_param['un_norm_Transition_matrix'],dim=1).detach())
    np.savetxt('../data/syntheticLR/LR_',np.array2string(LR),'Watson_HMM_comp0mu.csv',watsonhmm_param['emission_model_0']['mu'].detach())
    np.savetxt('../data/syntheticLR/LR_',np.array2string(LR),'Watson_HMM_comp0kappa.csv',watsonhmm_param['emission_model_0']['kappa'].detach())
    np.savetxt('../data/syntheticLR/LR_',np.array2string(LR),'Watson_HMM_comp1mu.csv',watsonhmm_param['emission_model_1']['mu'].detach())
    np.savetxt('../data/syntheticLR/LR_',np.array2string(LR),'Watson_HMM_comp1kappa.csv',watsonhmm_param['emission_model_1']['kappa'].detach())
    np.savetxt('../data/syntheticLR/LR_',np.array2string(LR),'Watson_HMM_viterbi.csv',np.transpose(Watson_HMM_best_paths))
    np.savetxt('../data/syntheticLR/LR_',np.array2string(LR),'Watson_HMM_emissionprobs.csv',np.squeeze(Watson_HMM_emission_probs))
    np.savetxt('../data/syntheticLR/LR_',np.array2string(LR),'Watson_HMM_likelihood.csv',like_Watson_HMM.detach())

