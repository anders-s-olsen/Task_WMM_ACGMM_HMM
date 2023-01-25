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
print(data.shape) #needs to be subjects, time, dims
print(data.dtype)
print(torch.norm(data,dim=2))


# ## ACG mixture
# train model with diff learning rates, get best model 

# In[7]:


best_LR = 0.1
int_epoch = 1000

ACG_MM = TorchMixtureModel(distribution_object=ACG,K=2, dist_dim=3)
#ACG_HMM = HMM(num_states=2, observation_dim=3, emission_dist=ACG)
#Watson_MM = TorchMixtureModel(distribution_object=Watson,K=2, dist_dim=3)
#Watson_HMM = HMM(num_states=2, observation_dim=3, emission_dist=Watson)

ACG_MM_optimizer = optim.Adam(ACG_MM.parameters(), lr=best_LR)
ACG_MM_ll = train_hmm(ACG_MM, data=torch.squeeze(data), optimizer=ACG_MM_optimizer, num_epoch=int_epoch, keep_bar=False)

#ACG_HMM_optimizer = optim.Adam(ACG_HMM.parameters(), lr=best_LR)
#ACG_HMM_ll = train_hmm(ACG_HMM, data=data, optimizer=ACG_HMM_optimizer, num_epoch=int_epoch, keep_bar=False)

#Watson_MM_optimizer = optim.Adam(Watson_MM.parameters(), lr=best_LR)
#Watson_MM_ll = train_hmm(Watson_MM, data=torch.squeeze(data), optimizer=Watson_MM_optimizer, num_epoch=int_epoch, keep_bar=False)

#Watson_HMM_optimizer = optim.Adam(Watson_HMM.parameters(), lr=best_LR)
#Watson_HMM_ll = train_hmm(Watson_HMM, data=data, optimizer=Watson_HMM_optimizer, num_epoch=int_epoch, keep_bar=False)


# ## extract parameters from the best model (should be pi around 0.5)

# In[14]:


acgmm_param = get_param(ACG_MM)
ACG_MM_post = ACG_MM.posterior(torch.squeeze(data))
np.savetxt('../data/synthetic/ACG_MM_prior.csv',torch.nn.functional.softmax(acgmm_param['un_norm_pi'],dim=0).detach())
np.savetxt('../data/synthetic/ACG_MM_comp0.csv',acgmm_param['mix_comp_0'].detach())
np.savetxt('../data/synthetic/ACG_MM_comp1.csv',acgmm_param['mix_comp_1'].detach())
np.savetxt('../data/synthetic/ACG_MM_posterior.csv',np.transpose(ACG_MM_post.detach()))
exit()
acghmm_param = get_param(ACG_HMM)
ACG_HMM_best_paths, ACG_HMM_paths_probs, ACG_HMM_emission_probs = ACG_HMM.viterbi2(data)
np.savetxt('../data/synthetic/ACG_HMM_prior.csv',torch.nn.functional.softmax(acghmm_param['un_norm_priors'],dim=0).detach())
np.savetxt('../data/synthetic/ACG_HMM_T.csv',torch.nn.functional.softmax(acghmm_param['un_norm_Transition_matrix'],dim=1).detach())
np.savetxt('../data/synthetic/ACG_HMM_comp0.csv',acghmm_param['emission_model_0'].detach())
np.savetxt('../data/synthetic/ACG_HMM_comp1.csv',acghmm_param['emission_model_1'].detach())
np.savetxt('../data/synthetic/ACG_HMM_viterbi.csv',np.transpose(ACG_HMM_best_paths))
np.savetxt('../data/synthetic/ACG_HMM_emissionprobs.csv',np.squeeze(ACG_HMM_emission_probs))

watsonmm_param = get_param(Watson_MM)
Watson_MM_post = Watson_MM.posterior(torch.squeeze(data))
np.savetxt('../data/synthetic/Watson_MM_prior.csv',torch.nn.functional.softmax(watsonmm_param['un_norm_pi'],dim=0).detach())
np.savetxt('../data/synthetic/Watson_MM_comp0mu.csv',watsonmm_param['mix_comp_0']['mu'].detach())
np.savetxt('../data/synthetic/Watson_MM_comp0kappa.csv',watsonmm_param['mix_comp_0']['kappa'].detach())
np.savetxt('../data/synthetic/Watson_MM_comp1mu.csv',watsonmm_param['mix_comp_1']['mu'].detach())
np.savetxt('../data/synthetic/Watson_MM_comp1kappa.csv',watsonmm_param['mix_comp_1']['kappa'].detach())
np.savetxt('../data/synthetic/Watson_MM_posterior.csv',np.transpose(Watson_MM_post.detach()))

watsonhmm_param = get_param(Watson_HMM)
Watson_HMM_best_paths, Watson_HMM_paths_probs, Watson_HMM_emission_probs = Watson_HMM.viterbi2(data)
np.savetxt('../data/synthetic/Watson_HMM_prior.csv',torch.nn.functional.softmax(watsonhmm_param['un_norm_priors'],dim=0).detach())
np.savetxt('../data/synthetic/Watson_HMM_T.csv',torch.nn.functional.softmax(watsonhmm_param['un_norm_Transition_matrix'],dim=1).detach())
np.savetxt('../data/synthetic/Watson_HMM_comp0mu.csv',watsonhmm_param['emission_model_0']['mu'].detach())
np.savetxt('../data/synthetic/Watson_HMM_comp0kappa.csv',watsonhmm_param['emission_model_0']['kappa'].detach())
np.savetxt('../data/synthetic/Watson_HMM_comp1mu.csv',watsonhmm_param['emission_model_1']['mu'].detach())
np.savetxt('../data/synthetic/Watson_HMM_comp1kappa.csv',watsonhmm_param['emission_model_1']['kappa'].detach())
np.savetxt('../data/synthetic/Watson_HMM_viterbi.csv',np.transpose(Watson_HMM_best_paths))
np.savetxt('../data/synthetic/Watson_HMM_emissionprobs.csv',np.squeeze(Watson_HMM_emission_probs))

