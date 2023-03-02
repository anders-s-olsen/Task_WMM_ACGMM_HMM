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
from src.various.training import train_hmm,train_hmm_batch

torch.set_num_threads(16)

def get_param(model, show=False):
    para = model.get_model_param()
    
    if show:
        for p_k in para:
            print(p_k)
            print(para[p_k])
            print(10*'---')
    
    return para

def run_experiment(m,r):
    num_repsouter = 1
    num_repsinner = 1
    num_regions = 100
    K=4

    datah5 = h5py.File('../data/processed/dataset_all_subjects_LEiDA_100.hdf5', 'r')
    #print(len(datah5.keys()))
    data_train = torch.zeros((29,240,num_regions),dtype=torch.double)
    data_train_W = torch.zeros((29,240,num_regions))
    for idx,subject in enumerate(list(datah5.keys())):
        data_train[idx] = torch.DoubleTensor(torch.tensor(datah5[subject]))
        data_train_W[idx] = torch.tensor(datah5[subject])
            
    data_train_concat = torch.concatenate([data_train[sub] for sub in range(data_train.shape[0])])
    data_train_concat_W = torch.concatenate([data_train_W[sub] for sub in range(data_train_W.shape[0])])
    #for m in range(4):
    if m==0:
        for D in np.arange(15):
            print(D)
            if D==0:
                model5 = TorchMixtureModel(distribution_object=Watson,K=K, dist_dim=data_train.shape[2])
                optimizer = optim.Adam(model5.parameters(), lr=0.1)
                like_Watson = train_hmm_batch(model5, data=data_train_W, optimizer=optimizer, num_epoch=25000, keep_bar=False,early_stopping=False,modeltype=0)
                param = get_param(model5)
                init = {}
                init['pi'] = param['un_norm_pi']
                init['comp'] = torch.zeros((K,data_train.shape[2],1),dtype=torch.double)
                for kk in range(K):
                    init['comp'][kk] = torch.sqrt(param['mix_comp_'+str(kk)]['kappa'])*torch.unsqueeze(param['mix_comp_'+str(kk)]['mu'],dim=1)


                model0 = TorchMixtureModel(distribution_object=ACG_lowrank,K=K, D=D+1,dist_dim=data_train.shape[2],init=init)
            else:
                model0 = TorchMixtureModel(distribution_object=ACG_lowrank,K=K, D=D+1,dist_dim=data_train.shape[2],init=init)
            optimizer = optim.Adam(model0.parameters(), lr=0.1)
            like = train_hmm_batch(model0, data=data_train, optimizer=optimizer, num_epoch=1000, keep_bar=False,early_stopping=False,modeltype=0)
            param = get_param(model0)
            init = {}
            init['pi'] = param['un_norm_pi']
            init['comp'] = torch.zeros((K,data_train.shape[2],D+1),dtype=torch.double)
            for kk in range(K):
                init['comp'][kk] = param['mix_comp_'+str(kk)]
        
    elif m==1:
        
        for D in np.arange(15):
            print(D)
            if D==0:
                model1 = HMM(emission_dist=ACG_lowrank,K=K, D=D+1,observation_dim=data_train.shape[2])
            else:
                model1 = HMM(emission_dist=ACG_lowrank,K=K, D=D+1,observation_dim=data_train.shape[2],init=init)
            optimizer = optim.Adam(model1.parameters(), lr=0.1)
            like = train_hmm_batch(model1, data=data_train, optimizer=optimizer, num_epoch=1000, keep_bar=False,early_stopping=False,modeltype=1)
            param = get_param(model1)
            init = {}
            init['pi'] = param['un_norm_priors']
            init['T'] = param['un_norm_Transition_matrix']
            init['comp'] = torch.zeros((K,data_train.shape[2],D+1),dtype=torch.double)
            for kk in range(K):
                init['comp'][kk] = param['emission_model_'+str(kk)]

    elif m==2:
        model2 = TorchMixtureModel(distribution_object=Watson,K=K, dist_dim=num_regions)
        optimizer = optim.Adam(model2.parameters(), lr=0.1)
        like = train_hmm_batch(model2, data=data_train_W, optimizer=optimizer, num_epoch=25000, keep_bar=False,early_stopping=False,modeltype=0)
    elif m==3:
        model3 = HMM(K=K, observation_dim=num_regions, emission_dist=Watson)
        optimizer = optim.Adam(model3.parameters(), lr=0.1)
        like = train_hmm_batch(model3, data=data_train_W, optimizer=optimizer, num_epoch=25000, keep_bar=False,early_stopping=False,modeltype=1)

    
    #plt.figure(),plt.plot(like)
    #R = torch.zeros((K,num_regions,num_regions))
    #A = torch.zeros((K,num_regions,num_regions))
    #for kk in range(K):
    #    R[kk] = param['mix_comp_'+str(kk)]@param['mix_comp_'+str(kk)].T
    #    A[kk] = torch.linalg.pinv(R[kk])
    #for kk in range(K):
    #    plt.figure(),plt.imshow(A[kk]),plt.colorbar()
    if m==0:
        param = get_param(model0)
        post = model0.posterior(data_train_concat)
        np.savetxt('../data/real_fit2/K'+str(K)+'ACG_MM_likelihood'+str(r)+'.csv',like)
        np.savetxt('../data/real_fit2/K'+str(K)+'ACG_MM_assignment'+str(r)+'.csv',np.transpose(post.detach()))
        np.savetxt('../data/real_fit2/K'+str(K)+'ACG_MM_prior'+str(r)+'.csv',torch.nn.functional.softmax(param['un_norm_pi'],dim=0).detach())
        for kk in range(K):
            #np.savetxt('../data/real_fit2/K'+str(K)+'ACG_MM_comp'+str(kk)+'_'+str(r)+'.csv',torch.linalg.pinv(param['mix_comp_'+str(kk)]@param['mix_comp_'+str(kk)].T).detach())
            np.savetxt('../data/real_fit2/K'+str(K)+'ACG_MM_comp'+str(kk)+'_'+str(r)+'.csv',param['mix_comp_'+str(kk)].detach())
    elif m==1:
        param = get_param(model1)
        best_path,xx,xxx = model1.viterbi2(data_train)
        np.savetxt('../data/real_fit2/K'+str(K)+'ACG_HMM_likelihood'+str(r)+'.csv',like)
        np.savetxt('../data/real_fit2/K'+str(K)+'ACG_HMM_assignment'+str(r)+'.csv',np.transpose(best_path))
        np.savetxt('../data/real_fit2/K'+str(K)+'ACG_HMM_prior'+str(r)+'.csv',torch.nn.functional.softmax(param['un_norm_priors'],dim=0).detach())
        np.savetxt('../data/real_fit2/K'+str(K)+'ACG_HMM_T'+str(r)+'.csv',torch.nn.functional.softmax(param['un_norm_Transition_matrix'],dim=1).detach())
        for kk in range(K):
            #np.savetxt('../data/real_fit2/K'+str(K)+'ACG_HMM_comp'+str(kk)+'_'+str(r)+'.csv',torch.linalg.pinv(param['emission_model_'+str(kk)]@param['emission_model_'+str(kk)].T).detach())
            np.savetxt('../data/real_fit2/K'+str(K)+'ACG_HMM_comp'+str(kk)+'_'+str(r)+'.csv',param['emission_model_'+str(kk)].detach())
    elif m==2:
        param = get_param(model2)
        post = model2.posterior(data_train_concat_W)
        np.savetxt('../data/real_fit2/K'+str(K)+'Watson_MM_likelihood'+str(r)+'.csv',like)
        np.savetxt('../data/real_fit2/K'+str(K)+'Watson_MM_assignment'+str(r)+'.csv',np.transpose(post.detach()))
        np.savetxt('../data/real_fit2/K'+str(K)+'Watson_MM_prior'+str(r)+'.csv',torch.nn.functional.softmax(param['un_norm_pi'],dim=0).detach())
        for kk in range(K):
            np.savetxt('../data/real_fit2/K'+str(K)+'Watson_MM_comp'+str(kk)+'_mu'+str(r)+'.csv',param['mix_comp_'+str(kk)]['mu'].detach())
            np.savetxt('../data/real_fit2/K'+str(K)+'Watson_MM_comp'+str(kk)+'_kappa'+str(r)+'.csv',param['mix_comp_'+str(kk)]['kappa'].detach())
    elif m==3:
        param = get_param(model3)
        best_path,xx,xxx = model3.viterbi2(data_train_W)
        np.savetxt('../data/real_fit2/K'+str(K)+'Watson_HMM_likelihood'+str(r)+'.csv',like)
        np.savetxt('../data/real_fit2/K'+str(K)+'Watson_HMM_assignment'+str(r)+'.csv',np.transpose(best_path))
        np.savetxt('../data/real_fit2/K'+str(K)+'Watson_HMM_prior'+str(r)+'.csv',torch.nn.functional.softmax(param['un_norm_priors'],dim=0).detach())
        np.savetxt('../data/real_fit2/K'+str(K)+'Watson_HMM_T'+str(r)+'.csv',torch.nn.functional.softmax(param['un_norm_Transition_matrix'],dim=1).detach())
        for kk in range(K):
            np.savetxt('../data/real_fit2/K'+str(K)+'Watson_HMM_comp'+str(kk)+'_mu'+str(r)+'.csv',param['emission_model_'+str(kk)]['mu'].detach())
            np.savetxt('../data/real_fit2/K'+str(K)+'Watson_HMM_comp'+str(kk)+'_kappa'+str(r)+'.csv',param['emission_model_'+str(kk)]['kappa'].detach())
if __name__=="__main__":
    #run_experiment(m=int(sys.argv[1]),r=int(sys.argv[2]))
    run_experiment(m=0,r=0)