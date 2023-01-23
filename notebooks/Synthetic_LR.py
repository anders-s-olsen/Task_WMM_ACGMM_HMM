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

# In[12]:


eval_LR = np.arange(0.1, 1, 0.1)
int_epoch = 100
acg_mm_LR_results = np.zeros((len(eval_LR),int_epoch))
acg_hmm_LR_results = np.zeros((len(eval_LR),int_epoch))
watson_mm_LR_results = np.zeros((len(eval_LR),int_epoch))
watson_hmm_LR_results = np.zeros((len(eval_LR),int_epoch))

for idx, LR in enumerate(tqdm(eval_LR)):
    #
    ACG_MM = TorchMixtureModel(distribution_object=ACG,K=2, dist_dim=3)
    ACG_HMM = HMM(num_states=2, observation_dim=3, emission_dist=ACG)
    Watson_MM = TorchMixtureModel(distribution_object=Watson,K=2, dist_dim=3)
    Watson_HMM = HMM(num_states=2, observation_dim=3, emission_dist=Watson)
    
    ACG_MM_optimizer = optim.Adam(ACG_MM.parameters(), lr=LR)
    ACG_MM_ll = train_hmm(ACG_MM, data=torch.squeeze(data), optimizer=ACG_MM_optimizer, num_epoch=int_epoch, keep_bar=False)
    acg_mm_LR_results[idx] = ACG_MM_ll
    
    ACG_HMM_optimizer = optim.Adam(ACG_HMM.parameters(), lr=LR)
    ACG_HMM_ll = train_hmm(ACG_HMM, data=data, optimizer=ACG_HMM_optimizer, num_epoch=int_epoch, keep_bar=False)
    acg_hmm_LR_results[idx] = ACG_HMM_ll
    
    Watson_MM_optimizer = optim.Adam(Watson_MM.parameters(), lr=LR)
    Watson_MM_ll = train_hmm(Watson_MM, data=torch.squeeze(data), optimizer=Watson_MM_optimizer, num_epoch=int_epoch, keep_bar=False)
    watson_mm_LR_results[idx] = Watson_MM_ll
    
    Watson_HMM_optimizer = optim.Adam(Watson_HMM.parameters(), lr=LR)
    Watson_HMM_ll = train_hmm(Watson_HMM, data=data, optimizer=Watson_HMM_optimizer, num_epoch=int_epoch, keep_bar=False)
    watson_hmm_LR_results[idx] = Watson_HMM_ll
    


# ## plot of learning rates

# In[7]:


plt.close()
fig, axs = plt.subplots(2, 2,figsize=(15, 15))

for ax in axs:
    axs[ax].xlabel('Epochs')
    axs[ax].ylabel('Log-Likelihood')
    axs[ax].legend(np.round(eval_LR, 3), ncol=1, bbox_to_anchor=(1.01, 1.05), loc='upper left', borderaxespad=0.)
    axs[ax].grid()


axs[0].plot(acg_mm_LR_results.T)
axs[0].title('Learning rate - ACG - Synthetic3D - mixture model')

axs[0].plot(acg_hmm_LR_results.T)
axs[0].title('Learning rate - ACG - Synthetic3D - Hidden Markov Model')

axs[0].plot(watson_mm_LR_results.T)
axs[0].title('Learning rate - Watson - Synthetic3D - mixture model')

axs[0].plot(watson_hmm_LR_results.T)
axs[0].title('Learning rate - Watson - Synthetic3D - Hidden Markov Model')

plt.savefig('../reports/synthetic_methods/synthetic_LRs.png')


# ## get best models

# In[11]:


#acg_mm_idx = np.argmax(acg_mm_LR_results)
#acg_hmm_idx = np.argmax(acg_hmm_LR_results)
#watson_mm_idx = np.argmax(watson_mm_LR_results)
#watson_hmm_idx = np.argmax(watson_hmm_LR_results)


# ## get emission probs and viterbi

# In[ ]:


#best_paths, paths_probs, emission_probs = best_model.viterbi2(data)
#np.savetxt('../data/synthetic/emissionprobs_ACG.csv', emission_probs, delimiter=',')
#np.savetxt('../data/synthetic/best_path_ACG.csv', best_paths, delimiter=',')


# ## extract parameters from the best model (should be pi=0.5)

# In[12]:


#acgbest_param = get_param(best_model)
#learned_sigma = torch.stack([acgbest_param[f'emission_model_{idx}'] for idx in range(best_model.N)])
#learned_pi = acgbest_param['un_norm_priors']
#learned_pi = torch.nn.functional.softmax(learned_pi,dim=0)
#print(learned_pi)

