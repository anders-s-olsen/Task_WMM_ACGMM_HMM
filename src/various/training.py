import torch
import numpy as np

from tqdm import tqdm

# Train Mixture model
def train_mixture(MixtureModel, data, optimizer, num_epoch=100, keep_bar=True):
    device = 'cpu'
    model = MixtureModel.to(device).train()

    epoch_likelihood_collector = np.zeros(num_epoch)
    for epoch in tqdm(range(num_epoch), leave=keep_bar):

        leida_vectors = data.to(device)

        NegativeLogLikelihood = -model(leida_vectors)  # OBS! Negative

        optimizer.zero_grad()
        NegativeLogLikelihood.backward()
        optimizer.step()

        epoch_likelihood_collector[epoch] = NegativeLogLikelihood.item()


    return epoch_likelihood_collector


def train_mixture_batches(MixtureModel, data, optimizer, num_epoch=100, keep_bar=True):
    device = 'cpu'
    model = MixtureModel.to(device).train()
    #print(device)
    epoch_likelihood_collector = np.zeros(num_epoch)

    for epoch in tqdm(range(num_epoch), leave=keep_bar):
        epoch_LogLikelihood = 0
        for subject in data:
            all_leida_vectors = subject.to(device)

            subject_Likelihood = -model(all_leida_vectors)  # OBS! Negative

            epoch_LogLikelihood += subject_Likelihood
            optimizer.zero_grad()
            subject_Likelihood.backward()
            optimizer.step()

        epoch_likelihood_collector[epoch] = epoch_LogLikelihood



    return epoch_likelihood_collector


# Train Hidden Markov model



def train_hmm(HMM, data, optimizer, num_epoch=100, keep_bar=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HMM.to(device).train()

    epoch_likelihood_collector = np.zeros(num_epoch)

    for epoch in tqdm(range(num_epoch), leave=keep_bar):

        subject_leida_vectors = data.to(device)

        NegativeLogLikelihood = -model(subject_leida_vectors)  # OBS! Negative

        optimizer.zero_grad()
        NegativeLogLikelihood.backward()
        optimizer.step()

        epoch_likelihood_collector[epoch] = NegativeLogLikelihood
        if epoch==0:
            torch.save(model.state_dict(),'../data/interim/model_checkpoint.pt')
            best_like = epoch_likelihood_collector[epoch]
        elif np.isin(epoch,np.linspace(0,num_epoch,int(num_epoch/5+1))):
            if epoch_likelihood_collector[epoch]<epoch_likelihood_collector[epoch-5] and epoch_likelihood_collector[epoch]<best_like:
                torch.save(model.state_dict(),'../data/interim/model_checkpoint.pt')
                np.savetxt('../data/interim/likelihood.txt',np.array((epoch,epoch_likelihood_collector[epoch])))
                best_like = epoch_likelihood_collector[epoch]

    return epoch_likelihood_collector


def train_hmm_subjects(HMM, data, optimizer, num_epoch=100, print_progress=False):
    model = HMM.to(device).train()

    epoch_likelihood_collector = np.zeros(num_epoch)

    for epoch in tqdm(range(num_epoch)):

        epoch_LogLikelihood = 0
        for subject in data:
            subejct_leida_vectors_seq = subject.to(device)

            subject_Likelihood = -model(subejct_leida_vectors_seq)  # OBS! Negative

            epoch_LogLikelihood += subject_Likelihood
            optimizer.zero_grad()
            subject_Likelihood.backward()
            optimizer.step()

        epoch_likelihood_collector[epoch] = epoch_LogLikelihood


        subject_leida_vectors = data.to(device)

        NegativeLogLikelihood = -model(subject_leida_vectors)  # OBS! Negative

        optimizer.zero_grad()
        NegativeLogLikelihood.backward()
        optimizer.step()

        epoch_likelihood_collector[epoch] = NegativeLogLikelihood
        raise NotImplementedError
    return epoch_likelihood_collector











if __name__ == '__main__':
    pass
