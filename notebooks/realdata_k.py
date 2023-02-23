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
    int_epoch = 100
    num_comp = np.arange(3,11)
    num_regions = 100
    sub=0

    datah5 = h5py.File('../data/processed/dataset_all_subjects_LEiDA_100.hdf5', 'r')
    #print(len(datah5.keys()))
    data_train = torch.zeros((29,120,num_regions))
    data_test = torch.zeros((29,120,num_regions))
    for idx,subject in enumerate(list(datah5.keys())):
        data_train[idx] = torch.tensor(datah5[subject][0:120])
        data_test[idx] = torch.tensor(datah5[subject][120:])
            
    data_train_concat = torch.concatenate([data_train[sub] for sub in range(data_train.shape[0])])
    data_test_concat = torch.concatenate([data_test[sub] for sub in range(data_test.shape[0])])
    #for m in range(4):
    for m in range(4):
        for r in range(num_repsouter):
            thres_like = 1000000000000000
            for r2 in range(num_repsinner):
                if m==0:
                    model = TorchMixtureModel(distribution_object=ACG,K=K, dist_dim=data_train.shape[2],regu=1e-2)
                    optimizer = optim.Adam(model.parameters(), lr=0.01)
                    like,model,like_best = train_hmm(model, data=data_train_concat, optimizer=optimizer, num_epoch=int_epoch, keep_bar=False,early_stopping=True)
                    test_like = -model.log_likelihood_mixture(data_test_concat)

                    param = get_param(model)
                    A = torch.zeros((K,data_train.shape[2],data_train.shape[2]))
                    for kk in range(K):
                        A[kk] = torch.linalg.pinv(param['mix_comp_'+str(kk)]@param['mix_comp_'+str(kk)].T)
                        np.savetxt('../data/real_K/K'+str(K)+'ACG_MM_comp'+str(kk)+'_'+str(r)+'.csv',A[kk].detach())
                    np.savetxt('../data/real_K/K'+str(K)+'ACG_MM_likelihood'+str(r)+'.csv',like_best)
                elif m==1:
                    model = HMM(num_states=K, observation_dim=data_train.shape[2], emission_dist=ACG,regu=1e-2)
                    optimizer = optim.Adam(model.parameters(), lr=0.1)
                    like,model,like_best = train_hmm(model, data=data_train, optimizer=optimizer, num_epoch=int_epoch, keep_bar=False,early_stopping=True)
                    test_like = -model.forward(data_test)
                    param = get_param(model)
                    A = torch.zeros((K,data_train.shape[2],data_train.shape[2]))
                    for kk in range(K):
                        A[kk] = torch.linalg.pinv(param['emission_model_'+str(kk)]@param['emission_model_'+str(kk)].T)
                        np.savetxt('../data/real_K/K'+str(K)+'ACG_HMM_comp'+str(kk)+'_'+str(r)+'.csv',A[kk].detach())
                    np.savetxt('../data/real_K/K'+str(K)+'ACG_HMM_likelihood'+str(r)+'.csv',like_best)
                elif m==2:
                    model = TorchMixtureModel(distribution_object=Watson,K=K, dist_dim=data_train.shape[2])
                    optimizer = optim.Adam(model.parameters(), lr=0.01)
                    like,model,like_best = train_hmm(model, data=data_train_concat, optimizer=optimizer, num_epoch=int_epoch, keep_bar=False,early_stopping=True)
                    test_like = -model.log_likelihood_mixture(data_test_concat)
                    np.savetxt('../data/real_K/K'+str(K)+'Watson_MM_likelihood'+str(r)+'.csv',like_best)
                    for kk in range(K):
                        np.savetxt('../data/real_K/K'+str(K)+'Watson_MM_comp'+str(kk)+'_mu'+str(r)+'.csv',param['mix_comp_'+str(kk)]['mu'].detach())
                        np.savetxt('../data/real_K/K'+str(K)+'Watson_MM_comp'+str(kk)+'_kappa'+str(r)+'.csv',param['mix_comp_'+str(kk)]['kappa'].detach())
                elif m==3:
                    model = HMM(num_states=K, observation_dim=data_train.shape[2], emission_dist=Watson)
                    optimizer = optim.Adam(model.parameters(), lr=0.01)
                    like,model,like_best = train_hmm(model, data=data_train, optimizer=optimizer, num_epoch=int_epoch, keep_bar=False,early_stopping=True)
                    test_like = -model.forward(data_test)
                    np.savetxt('../data/real_K/K'+str(K)+'Watson_HMM_likelihood'+str(r)+'.csv',like_best)
                    for kk in range(K):
                        np.savetxt('../data/real_K/K'+str(K)+'Watson_HMM_comp'+str(kk)+'_mu'+str(r)+'.csv',param['emission_model_'+str(kk)]['mu'].detach())
                        np.savetxt('../data/real_K/K'+str(K)+'Watson_HMM_comp'+str(kk)+'_kappa'+str(r)+'.csv',param['emission_model_'+str(kk)]['kappa'].detach())
                
                continue


                # load best model and calculate posterior or viterbi
                #model.load_state_dict(torch.load('../data/interim/model_checkpoint.pt'))
                #like_best = np.loadtxt('../data/interim/likelihood.txt')
                if like_best[1]<thres_like:
                    thres_like = like_best[1]
                    param = get_param(model)
                    plt.figure(),plt.plot(like)
                    R = torch.zeros((K,data_train.shape[2],data_train.shape[2]))
                    A = torch.zeros((K,data_train.shape[2],data_train.shape[2]))
                    for kk in range(K):
                        R[kk] = param['mix_comp_'+str(kk)]@param['mix_comp_'+str(kk)].T
                        A[kk] = torch.linalg.pinv(R[kk])
                        np.savetxt('../data/nodes_edges/K'+str(K)+'ACG_MM_comp'+str(kk)+'_'+str(r)+'.csv',A[kk].detach())
                    for kk in range(K):
                        plt.figure(),plt.imshow(A[kk]),plt.colorbar()
                    if m==0:
                        post = model.posterior(data_concat)
                        np.savetxt('../data/real_K/K'+str(K)+'ACG_MM_likelihood'+str(r)+'.csv',like_best)
                        np.savetxt('../data/real_K/K'+str(K)+'ACG_MM_assignment'+str(r)+'.csv',np.transpose(post.detach()))
                        np.savetxt('../data/real_K/K'+str(K)+'ACG_MM_prior'+str(r)+'.csv',torch.nn.functional.softmax(param['un_norm_pi'],dim=0).detach())
                        for kk in range(K):
                            np.savetxt('../data/real_K/K'+str(K)+'ACG_MM_comp'+str(kk)+'_'+str(r)+'.csv',param['mix_comp_'+str(kk)].detach())
                    elif m==1:
                        best_path,xx,xxx = model.viterbi2(data_train)
                        np.savetxt('../data/real_K/K'+str(K)+'ACG_HMM_likelihood'+str(r)+'.csv',like_best)
                        np.savetxt('../data/real_K/K'+str(K)+'ACG_HMM_assignment'+str(r)+'.csv',np.transpose(best_path))
                        np.savetxt('../data/real_K/K'+str(K)+'ACG_HMM_prior'+str(r)+'.csv',torch.nn.functional.softmax(param['un_norm_priors'],dim=0).detach())
                        np.savetxt('../data/real_K/K'+str(K)+'ACG_HMM_T'+str(r)+'.csv',torch.nn.functional.softmax(param['un_norm_Transition_matrix'],dim=1).detach())
                        for kk in range(K):
                            np.savetxt('../data/real_K/K'+str(K)+'ACG_HMM_comp'+str(kk)+'_'+str(r)+'.csv',param['emission_model_'+str(kk)].detach())
                    elif m==2:
                        post = model.posterior(data_concat)
                        np.savetxt('../data/real_K/K'+str(K)+'Watson_MM_likelihood'+str(r)+'.csv',like_best)
                        np.savetxt('../data/real_K/K'+str(K)+'Watson_MM_assignment'+str(r)+'.csv',np.transpose(post.detach()))
                        np.savetxt('../data/real_K/K'+str(K)+'Watson_MM_prior'+str(r)+'.csv',torch.nn.functional.softmax(param['un_norm_pi'],dim=0).detach())
                        for kk in range(K):
                            np.savetxt('../data/real_K/K'+str(K)+'Watson_MM_comp'+str(kk)+'_mu'+str(r)+'.csv',param['mix_comp_'+str(kk)]['mu'].detach())
                            np.savetxt('../data/real_K/K'+str(K)+'Watson_MM_comp'+str(kk)+'_kappa'+str(r)+'.csv',param['mix_comp_'+str(kk)]['kappa'].detach())
                    elif m==3:
                        best_path,xx,xxx = model.viterbi2(data)
                        np.savetxt('../data/real_K/K'+str(K)+'Watson_HMM_likelihood'+str(r)+'.csv',like_best)
                        np.savetxt('../data/real_K/K'+str(K)+'Watson_HMM_assignment'+str(r)+'.csv',np.transpose(best_path))
                        np.savetxt('../data/real_K/K'+str(K)+'Watson_HMM_prior'+str(r)+'.csv',torch.nn.functional.softmax(param['un_norm_priors'],dim=0).detach())
                        np.savetxt('../data/real_K/K'+str(K)+'Watson_HMM_T'+str(r)+'.csv',torch.nn.functional.softmax(param['un_norm_Transition_matrix'],dim=1).detach())
                        for kk in range(K):
                            np.savetxt('../data/real_K/K'+str(K)+'Watson_HMM_comp'+str(kk)+'_mu'+str(r)+'.csv',param['emission_model_'+str(kk)]['mu'].detach())
                            np.savetxt('../data/real_K/K'+str(K)+'Watson_HMM_comp'+str(kk)+'_kappa'+str(r)+'.csv',param['emission_model_'+str(kk)]['kappa'].detach())
if __name__=="__main__":
    run_experiment(K=int(sys.argv[1]))
    #run_experiment(m=3)