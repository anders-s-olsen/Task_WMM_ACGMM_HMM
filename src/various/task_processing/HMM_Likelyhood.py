#%%
import torch
import numpy as np
import torch

Softmax = torch.nn.Softmax(0)
Softplus = torch.nn.Softplus()

# Logarithmic stable sum for a matrix over rows
@torch.jit.script
def lsumMatrix(X):
    Max = X.max(1).values
    return torch.log(torch.exp((X.T-Max).T).sum(1)) + Max

# Logarithmic stable sum 
@torch.jit.script
def lsum(x):
    return x.max() + torch.log(torch.exp(x-x.max()).sum())

# Initilize parameters of Hidden Markov Model
def InitializeParameters(n,p,K):
    mu = torch.zeros((p,K))
    for j in range(K):
            val = 1 if j % 2 == 0 else -1
            mu[(j*int(p/K)),j] = val

    # Intialize pi,mu and kappa
    grad = True
    pi = torch.tensor([1/K for _ in range(K)],requires_grad=grad)
    kappa = torch.tensor([1. for _ in range(K)],requires_grad=grad)  
    mu.requires_grad = grad
    Tk = torch.ones((n,K,K),requires_grad=grad)
    Pinit = torch.ones((K,n),requires_grad=grad)

    return kappa,mu,Tk,Pinit

# Initilize parameters of Hidden Markov Model user farthest first
def InitializeParametersFF(X_tensor,n,p,K):
    mu = torch.zeros(p,K)
    observations = X_tensor.shape[1]
    mu[:,0] = X_tensor[:,np.random.randint(observations)]
    for new in range(1,K):
        mu[:,new]=X_tensor[:,np.argmin((((X_tensor.T @ mu)[:,0:new])**2).max(1).values)]

    # Intialize pi,mu and kappa
    grad = True
    pi = torch.tensor([1/K for _ in range(K)],requires_grad=grad)
    kappa = torch.tensor([1. for _ in range(K)],requires_grad=grad)  
    mu.requires_grad = grad
    Tk = torch.ones((n,K,K),requires_grad=grad)
    Pinit = torch.ones((K,n),requires_grad=grad)

    return kappa,mu,Tk,Pinit



# Log - kummers function
@torch.jit.script
def M_log(a : float ,c : float,k):
    
    M0 = torch.ones(len(k))
    Madd = torch.ones(len(k))
    M0_log = torch.zeros(len(k))


    for j in range(1,100000):
        Madd = Madd * (a+j-1)/(c+j-1) * k/j
        M0 += Madd
        Madd_log = torch.log(1+Madd.clone()/M0.clone())
        M0_log += Madd_log
        if all(Madd_log < 1e-10) :
            break
    return M0_log

# pdf normalizer on log-scale
@torch.jit.script
def log_c(p: int,kappa):
        return torch.lgamma(p/2) - torch.log(torch.tensor(2 * np.pi**(p/2))) - M_log(1/2,p/2,kappa) 

# log-pdf watson distribution
@torch.jit.script
def log_pdf(X,mu,kappa,p : int):
        Wp = log_c(p,kappa) + kappa * (mu.T @ X).T**2
        return Wp

# Likelihood of trajectory of one subject given parameters
@torch.jit.script
def HMM_log_likelihood(X,Pinit,kappa,mu,Tk,p : int ,K : int):

    Tlog = torch.log(Tk)    
    Emmision_Prop = log_pdf(X,mu,kappa,p).T

    Prob = torch.log(Pinit) + Emmision_Prop[:,0]
    for n in range(1,330):
        Prob = lsumMatrix(Prob.clone() + Tlog) + Emmision_Prop[:,n]
        
    return lsum(Prob) 

# Sum over likelihood for each subject
@torch.jit.script
def Accumulated_HHM_LL(X,Pinit,kappa,mu,Tk,n : int,p : int,K : int):

    # Constraining Parameters:
    kappa_con = torch.minimum(Softplus(kappa),torch.tensor([800]))
    mu_con = mu /torch.sqrt((mu**2).sum(0))

    Subjectlog_Likelihood = torch.zeros(n)
    
    
    for subject in range(int(n)):
        Subjectlog_Likelihood[subject] = HMM_log_likelihood(X[:,330*subject:330*(subject+1)],Softmax(Pinit[:,subject]),kappa_con,mu_con,Softmax(Tk[subject]),p,K)
    
    return Subjectlog_Likelihood.sum()


# Caluclate log-likelihood of half a subject given parameters
@torch.jit.script
def HMM_log_likelihoodHalf(X,Pinit,kappa,mu,Tk,p : int ,K : int):

    Tlog = torch.log(Tk)    
    Emmision_Prop = log_pdf(X,mu,kappa,p).T

    Prob = torch.log(Pinit) + Emmision_Prop[:,0]
    for n in range(1,165):
        Prob = lsumMatrix(Prob.clone() + Tlog) + Emmision_Prop[:,n]
        
    return lsum(Prob) 



# Accumulate likelihood of each subject given parameters
@torch.jit.script
def Accumulated_HHM_LLHalf(X,Pinit,kappa,mu,Tk,n : int,p : int,K : int):

    # Constraining Parameters:
    kappa_con = torch.minimum(Softplus(kappa),torch.tensor([800]))
    mu_con = mu /torch.sqrt((mu**2).sum(0))

    Subjectlog_Likelihood = torch.zeros(n)
    
    
    for subject in range(int(n)):
        Subjectlog_Likelihood[subject] = HMM_log_likelihoodHalf(X[:,165*subject:165*(subject+1)],Softmax(Pinit[:,subject]),kappa_con,mu_con,Softmax(Tk[subject]),p,K)
    
    return Subjectlog_Likelihood.sum()



# Optimize inputparameters
def Optimizationloop(X,Parameters,lose,Optimizer,n,n_iters : int,p=90,K =7):
        Error_prev = 0
        for epoch in range(n_iters):
                Error = -lose(X,*Parameters,n,p,K=K)

                if torch.isnan(Error):
                    print("Optimizationloop has Converged")
                    break

                Error.backward()

                # Using optimzer
                Optimizer.step()
                Optimizer.zero_grad()

                #if abs(Error)-abs(Error_prev) < 1e-2:
                #    break

                #Error_prev = Error
                if epoch % 100 == 0:
                        print(f"epoch {epoch+1}; Log-Likelihood = {Error}")
        return Parameters

# Optimize inputparameters and return trajectory of log likelihoods
def OptimizationTraj(X,Parameters,lose,Optimizer,n,n_iters : int,p=90,K =7):
        Trajectory = np.zeros(n_iters)
        for epoch in range(n_iters):
                Error = -lose(X,*Parameters,n,p,K=K)

                if torch.isnan(Error):
                    print("Optimizationloop has Converged")
                    break

                Error.backward()

                # Using optimzer
                Optimizer.step()
                Optimizer.zero_grad()

                #if abs(Error)-abs(Error_prev) < 1e-2:
                #    break

                #Error_prev = Error
                if epoch % 100 == 0 or epoch == n_iters-1:
                        print(f"epoch {epoch+1}; Log-Likelihood = {Error}")
                Trajectory[epoch] = Error
        return Trajectory

