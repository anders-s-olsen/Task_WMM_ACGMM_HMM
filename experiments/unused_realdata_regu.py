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

from models.AngularCentralGauss_chol import AngularCentralGaussian as ACG
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

def run_experiment(m):
    print(m)
    num_repsouter = 1
    num_repsinner = 1
    int_epoch = 20000
    K=3
    data = torch.zeros((29,240,200))
    sub=0

    lambdas = [0,0.0001,0.001,0.01,0.1,1,10,100]

    datah5 = h5py.File('../data/processed/dataset_all_subjects_LEiDA.hdf5', 'r')
    #print(len(datah5.keys()))
    for idx,subject in enumerate(list(datah5.keys())):
        data[idx] = torch.tensor(datah5[subject])
            
    data_concat = torch.concatenate([data[sub] for sub in range(data.shape[0])])
#for m in range(4):
    for lambd in lambdas:
        for r in range(num_repsouter):
            thres_like = 1000000000000000
            for r2 in range(num_repsinner):
                if m==0:
                    model = TorchMixtureModel(distribution_object=ACG,K=K, dist_dim=data.shape[2],regu=lambd)
                    optimizer = optim.Adam(model.parameters(), lr=0.01)
                    like,model,like_best = train_hmm(model, data=data_concat, optimizer=optimizer, num_epoch=int_epoch, keep_bar=False,early_stopping=True)
                elif m==1:
                    model = HMM(num_states=K, observation_dim=data.shape[2], emission_dist=ACG,regu=lambd)
                    optimizer = optim.Adam(model.parameters(), lr=0.01)
                    like,model,like_best = train_hmm(model, data=data, optimizer=optimizer, num_epoch=int_epoch, keep_bar=False,early_stopping=True)
                
                # load best model and calculate posterior or viterbi
                #model.load_state_dict(torch.load('../data/interim/model_checkpoint.pt'))
                #like_best = np.loadtxt('../data/interim/likelihood.txt')
                if like_best[1]<thres_like:
                    thres_like = like_best[1]
                    param = get_param(model)

                    if m==0:
                        np.savetxt('../data/real_regu/ACG_MM_lambda_'+str(lambd)+'like',like)
                        plt.figure(),plt.plot(like)
                        plt.savefig('../data/real_regu/ACG_MM_lambda_'+str(lambd)+'like.png')
                        plt.figure(),plt.imshow(torch.linalg.inv(param['mix_comp_0']@param['mix_comp_0'].T))
                        plt.savefig('../data/real_regu/ACG_MM_lambda_'+str(lambd)+'comp0.png')
                        plt.figure(),plt.imshow(torch.linalg.inv(param['mix_comp_1']@param['mix_comp_1'].T))
                        plt.savefig('../data/real_regu/ACG_MM_lambda_'+str(lambd)+'comp1.png')
                        plt.figure(),plt.imshow(torch.linalg.inv(param['mix_comp_2']@param['mix_comp_2'].T))
                        plt.savefig('../data/real_regu/ACG_MM_lambda_'+str(lambd)+'comp2.png')
                    elif m==1:
                        np.savetxt('../data/real_regu/ACG_HMM_lambda_'+str(lambd)+'like',like)
                        plt.figure(),plt.plot(like),plt.show()
                        plt.savefig('../data/real_regu/ACG_HMM_lambda_'+str(lambd)+'like.png')
                        plt.figure(),plt.imshow(torch.linalg.inv(param['mix_comp_0']@param['mix_comp_0'].T)),plt.show()
                        plt.savefig('../data/real_regu/ACG_HMM_lambda_'+str(lambd)+'comp0.png')
                        plt.figure(),plt.imshow(torch.linalg.inv(param['mix_comp_1']@param['mix_comp_1'].T)),plt.show()
                        plt.savefig('../data/real_regu/ACG_HMM_lambda_'+str(lambd)+'comp1.png')
                        plt.figure(),plt.imshow(torch.linalg.inv(param['mix_comp_2']@param['mix_comp_2'].T)),plt.show()
                        plt.savefig('../data/real_regu/ACG_HMM_lambda_'+str(lambd)+'comp2.png')

if __name__=="__main__":
    #run_experiment(m=int(sys.argv[1]))
    run_experiment(m=0)