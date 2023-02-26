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
from src.various.training import train_hmm_batch, train_hmm,train_hmm_lbfgs,train_hmm_subject

torch.set_num_threads(16)

def get_param(model, show=True):
    para = model.get_model_param()
    
    if show:
        for p_k in para:
            print(p_k)
            print(para[p_k])
            print(10*'---')
    
    return para


def run_experiment(K):
    num_repsouter = 5
    num_repsinner = 1
    int_epoch = 100000
    num_regions = 100

    datah5 = h5py.File('../data/processed/dataset_all_subjects_LEiDA_100.hdf5', 'r')
    #print(len(datah5.keys()))
    data_train = torch.zeros((29,120,num_regions))
    data_test = torch.zeros((29,120,num_regions))
    for idx,subject in enumerate(list(datah5.keys())):
        data_train[idx] = torch.tensor(datah5[subject][0:120])
        data_test[idx] = torch.tensor(datah5[subject][120:])
            
    data_train_concat = torch.concatenate([data_train[sub] for sub in range(data_train.shape[0])])
    data_test_concat = torch.concatenate([data_test[sub] for sub in range(data_test.shape[0])])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #for m in range(4):
    for r in range(num_repsouter):
        for m in range(4):
            if m==2:
                model2 = TorchMixtureModel(distribution_object=Watson,K=K, dist_dim=data_train.shape[2])
                optimizer = optim.Adam(model2.parameters(), lr=0.01)
                like = train_hmm_batch(model2, data=data_train, optimizer=optimizer, num_epoch=int_epoch, keep_bar=False,early_stopping=False,modeltype=0)
                test_like = -model2.log_likelihood_mixture(data_test_concat.to(device)).cpu()
                np.savetxt('../data/real_K/K'+str(K)+'Watson_MM_testlikelihood'+str(r)+'.csv',np.array((test_like.detach(),K)))
                np.savetxt('../data/real_K/K'+str(K)+'Watson_MM_likelihood'+str(r)+'.csv',like)
                param = get_param(model2)
                for kk in range(K):
                    np.savetxt('../data/real_K/K'+str(K)+'Watson_MM_comp'+str(kk)+'_mu'+str(r)+'.csv',param['mix_comp_'+str(kk)]['mu'].detach())
                    np.savetxt('../data/real_K/K'+str(K)+'Watson_MM_comp'+str(kk)+'_kappa'+str(r)+'.csv',param['mix_comp_'+str(kk)]['kappa'].detach())
                post = model2.posterior(data_test_concat.to(device)).cpu()
                np.savetxt('../data/real_K/K'+str(K)+'Watson_MM_assignment'+str(r)+'.csv',np.transpose(post.detach()))
            elif m==3:
                model3 = HMM(K=K, observation_dim=data_train.shape[2], emission_dist=Watson)
                optimizer = optim.Adam(model3.parameters(), lr=0.01)
                like = train_hmm_batch(model3, data=data_train, optimizer=optimizer, num_epoch=int_epoch, keep_bar=False,early_stopping=False,modeltype=1)
                test_like = -model3.forward(data_test.to(device)).cpu()
                np.savetxt('../data/real_K/K'+str(K)+'Watson_HMM_testlikelihood'+str(r)+'.csv',np.array((test_like.detach(),K)))
                np.savetxt('../data/real_K/K'+str(K)+'Watson_HMM_likelihood'+str(r)+'.csv',like)
                param = get_param(model3)
                for kk in range(K):
                    np.savetxt('../data/real_K/K'+str(K)+'Watson_HMM_comp'+str(kk)+'_mu'+str(r)+'.csv',param['emission_model_'+str(kk)]['mu'].detach())
                    np.savetxt('../data/real_K/K'+str(K)+'Watson_HMM_comp'+str(kk)+'_kappa'+str(r)+'.csv',param['emission_model_'+str(kk)]['kappa'].detach())
                post,x,xx = model3.viterbi2(data_test.to(device)).cpu()
                np.savetxt('../data/real_K/K'+str(K)+'Watson_HMM_assignment'+str(r)+'.csv',np.transpose(post))
                
if __name__=="__main__":
    run_experiment(K=int(sys.argv[1]))
    #run_experiment(K=4)