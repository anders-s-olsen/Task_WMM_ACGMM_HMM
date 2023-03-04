import torch
import numpy as np

from tqdm import tqdm

##### Batch training

def train_hmm_batch(HMM, data, optimizer, num_epoch=100, keep_bar=True,early_stopping=False,modeltype=0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #print('Currently using device: ')
    #print(device)
    model = HMM.to(device).train()

    epoch_likelihood_collector = np.zeros(num_epoch)

    subject_leida_vectors = data.to(device)

    for epoch in tqdm(range(num_epoch), leave=keep_bar):

        indices = torch.randperm(data.shape[0])
        epoch_likelihood = 0
        for batch in range(6):
            if batch<5:
                idx = indices[:5]
                indices = indices[5:]
            else:
                idx = indices
            if modeltype==0:
                if len(idx)==0:
                    batch_likelihood = -model(torch.squeeze(subject_leida_vectors))
                else:
                    batch_likelihood = -model(torch.concatenate([subject_leida_vectors[idx][sub] for sub in range(len(idx))]))  # OBS! Negative
            else:
                batch_likelihood = -model(subject_leida_vectors[idx])  # OBS! Negative
            epoch_likelihood += batch_likelihood
            optimizer.zero_grad(set_to_none=True)
            batch_likelihood.backward()
            optimizer.step()

        epoch_likelihood_collector[epoch] = epoch_likelihood
        if early_stopping:
            if epoch==0:
                ident = torch.randint(0,10000000,(1,1)).detach()
                torch.save(model.state_dict(),'../data/interim/model_checkpoint'+str(ident)+'.pt')
                best_like = epoch_likelihood_collector[epoch]
            elif np.isin(epoch,np.linspace(0,num_epoch,int(num_epoch/5+1))):
                if epoch_likelihood_collector[epoch]<epoch_likelihood_collector[epoch-5] and epoch_likelihood_collector[epoch]<best_like:
                    torch.save(model.state_dict(),'../data/interim/model_checkpoint'+str(ident)+'.pt')
                    np.savetxt('../data/interim/likelihood'+str(ident)+'.txt',np.array((epoch,epoch_likelihood_collector[epoch])))
                    best_like = epoch_likelihood_collector[epoch]
                    like_best = np.array((epoch,epoch_likelihood_collector[epoch]))
    if early_stopping:
        model.load_state_dict(torch.load('../data/interim/model_checkpoint'+str(ident)+'.pt'))
        return epoch_likelihood_collector,model,like_best
    else: 
        return epoch_likelihood_collector


def train_hmm(HMM, data, optimizer, num_epoch=100, keep_bar=True,early_stopping=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Currently using device: ')
    print(device)
    model = HMM.to(device).train()

    epoch_likelihood_collector = np.zeros(num_epoch)

    for epoch in tqdm(range(num_epoch), leave=keep_bar):

        subject_leida_vectors = data.to(device)

        NegativeLogLikelihood = -model(subject_leida_vectors)  # OBS! Negative

        optimizer.zero_grad(set_to_none=True)
        NegativeLogLikelihood.backward()
        optimizer.step()

        epoch_likelihood_collector[epoch] = NegativeLogLikelihood
        if early_stopping:
            if epoch==0:
                ident = torch.randint(0,10000000,(1,1)).detach()
                torch.save(model.state_dict(),'../data/interim/model_checkpoint'+str(ident)+'.pt')
                best_like = epoch_likelihood_collector[epoch]
            elif np.isin(epoch,np.linspace(0,num_epoch,int(num_epoch/100+1))):
                if epoch_likelihood_collector[epoch]<epoch_likelihood_collector[epoch-5] and epoch_likelihood_collector[epoch]<best_like:
                    torch.save(model.state_dict(),'../data/interim/model_checkpoint'+str(ident)+'.pt')
                    np.savetxt('../data/interim/likelihood'+str(ident)+'.txt',np.array((epoch,epoch_likelihood_collector[epoch])))
                    best_like = epoch_likelihood_collector[epoch]
                    like_best = np.array((epoch,epoch_likelihood_collector[epoch]))


    if early_stopping:
        model.load_state_dict(torch.load('../data/interim/model_checkpoint'+str(ident)+'.pt'))
        return epoch_likelihood_collector,model,like_best
    else: 
        return epoch_likelihood_collector



def train_hmm_lbfgs(HMM, data, optimizer, num_epoch=100, keep_bar=True,early_stopping=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Currently using device: ')
    print(device)
    model = HMM.to(device).train()

    lbfgs = torch.optim.LBFGS(model.parameters())

    epoch_likelihood_collector = np.zeros(num_epoch)
    subject_leida_vectors = data.to(device)
    def closure():
            lbfgs.zero_grad()
            objective = -model(subject_leida_vectors)
            objective.backward()
            return objective
    for epoch in tqdm(range(num_epoch), leave=keep_bar):

        NegativeLogLikelihood = -model(subject_leida_vectors)  # OBS! Negative
        lbfgs.step(closure)

        epoch_likelihood_collector[epoch] = NegativeLogLikelihood
        if early_stopping:
            if epoch==0:
                ident = torch.randint(0,10000000,(1,1)).detach()
                torch.save(model.state_dict(),'../data/interim/model_checkpoint'+str(ident)+'.pt')
                best_like = epoch_likelihood_collector[epoch]
            elif np.isin(epoch,np.linspace(0,num_epoch,int(num_epoch/100+1))):
                if epoch_likelihood_collector[epoch]<epoch_likelihood_collector[epoch-5] and epoch_likelihood_collector[epoch]<best_like:
                    torch.save(model.state_dict(),'../data/interim/model_checkpoint'+str(ident)+'.pt')
                    np.savetxt('../data/interim/likelihood'+str(ident)+'.txt',np.array((epoch,epoch_likelihood_collector[epoch])))
                    best_like = epoch_likelihood_collector[epoch]
                    like_best = np.array((epoch,epoch_likelihood_collector[epoch]))


    if early_stopping:
        model.load_state_dict(torch.load('../data/interim/model_checkpoint'+str(ident)+'.pt'))
        return epoch_likelihood_collector,model,like_best
    else: 
        return epoch_likelihood_collector


if __name__ == '__main__':
    pass
