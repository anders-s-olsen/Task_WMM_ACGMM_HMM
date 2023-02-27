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
    num_repsouter = 5
    num_repsinner = 1
    int_epoch = 1000
    num_regions = 100

    datah5 = h5py.File('../data/processed/dataset_all_subjects_LEiDA_100.hdf5', 'r')
    #print(len(datah5.keys()))
    data_train = torch.zeros((29,120,num_regions),dtype=torch.double)
    data_test = torch.zeros((29,120,num_regions),dtype=torch.double)
    for idx,subject in enumerate(list(datah5.keys())):
        x = np.array(datah5[subject])
        data_train[idx] = torch.DoubleTensor(np.array(datah5[subject][0:120]))
        data_test[idx] = torch.DoubleTensor(np.array(datah5[subject][120:]))
            
    data_train_concat = torch.concatenate([data_train[sub] for sub in range(data_train.shape[0])])
    data_test_concat = torch.concatenate([data_test[sub] for sub in range(data_test.shape[0])])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #for m in range(4):
    for r in range(num_repsouter):
        for m in range(4):
            test_like = torch.zeros(15,dtype=torch.double)
            if m==0:
                for D in np.arange(15):
                    print(D)
                    if D==0:
                        model0 = TorchMixtureModel(distribution_object=ACG_lowrank,K=K, D=D+1,dist_dim=data_train.shape[2])
                    else:
                        model0 = TorchMixtureModel(distribution_object=ACG_lowrank,K=K, D=D+1,dist_dim=data_train.shape[2],init=init)
                    optimizer = optim.Adam(model0.parameters(), lr=0.1)
                    like = train_hmm_batch(model0, data=data_train, optimizer=optimizer, num_epoch=int_epoch, keep_bar=False,early_stopping=False,modeltype=0)
                    param = get_param(model0)
                    init = {}
                    init['pi'] = param['un_norm_pi']
                    init['comp'] = torch.zeros((K,data_train.shape[2],D+1),dtype=torch.double)
                    for kk in range(K):
                        init['comp'][kk] = param['mix_comp_'+str(kk)]
                    test_like[D] = -model0.log_likelihood_mixture(data_test_concat.to(device)).cpu()
                np.savetxt('../data/real_K/K'+str(K)+'ACG_MM_testlikelihood'+str(r)+'.csv',test_like.detach())
                np.savetxt('../data/real_K/K'+str(K)+'ACG_MM_likelihood'+str(r)+'.csv',like)
                param = get_param(model0)
                for kk in range(K):
                    np.savetxt('../data/real_K/K'+str(K)+'ACG_MM_comp'+str(kk)+'_mu'+str(r)+'.csv',param['mix_comp_'+str(kk)].detach())
                post = model0.posterior(data_test_concat.to(device))
                np.savetxt('../data/real_K/K'+str(K)+'ACG_MM_assignment'+str(r)+'.csv',np.transpose(post.cpu().detach()))
            elif m==1:
                for D in np.arange(50):
                    print(D)
                    if D==0:
                        model0 = HMM(K=K, emission_dist=ACG_lowrank, observation_dim=data_train.shape[2],D=D+1)
                    else:
                        model0 = HMM(K=K, emission_dist=ACG_lowrank, observation_dim=data_train.shape[2],D=D+1,init=init)
                    optimizer = optim.Adam(model0.parameters(), lr=0.1)
                    like = train_hmm_batch(model0, data=data_train, optimizer=optimizer, num_epoch=int_epoch, keep_bar=False,early_stopping=False,modeltype=1)
                    param = get_param(model0)
                    init = {}
                    init['pi'] = param['un_norm_priors']
                    init['T'] = param['un_norm_Transition_matrix']
                    init['comp'] = torch.zeros((K,data_train.shape[2],D+1),dtype=torch.double)
                    for kk in range(K):
                        init['comp'][kk] = param['emission_model_'+str(kk)]
                    test_like[D] = -model0.forward(data_test.to(device)).cpu()
                np.savetxt('../data/real_K/K'+str(K)+'ACG_HMM_testlikelihood'+str(r)+'.csv',test_like.detach())
                np.savetxt('../data/real_K/K'+str(K)+'ACG_HMM_likelihood'+str(r)+'.csv',like)
                param = get_param(model0)
                for kk in range(K):
                    np.savetxt('../data/real_K/K'+str(K)+'ACG_HMM_comp'+str(kk)+'_mu'+str(r)+'.csv',param['emission_model_'+str(kk)].detach())
                post,x,xx = model0.viterbi2(data_test.to(device))
                np.savetxt('../data/real_K/K'+str(K)+'ACG_HMM_assignment'+str(r)+'.csv',np.transpose(post.cpu()))
                
if __name__=="__main__":
    run_experiment(K=int(sys.argv[1]))
    #run_experiment(K=4)