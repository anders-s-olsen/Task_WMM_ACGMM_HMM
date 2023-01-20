#%%
from math import factorial
from scipy.special import poch
import numpy as np

def matrix_vector_product(a,b):
    M = np.zeros((90,90))
    for i in range(90):
        for j in range(90):
            M[i,j] = a[i]*b[j]
    return M

def faculty(x):
    if x==0: 
        return 1
    else: 
        return x*faculty(x-1)

def rising_factorial(a,j):
    if j == 0: 
        return 1
    else: 
        return (a+j-1)*rising_factorial(a,j-1)

# Class containing each function discribing a Watson Distribution
class WatsonDistribution:
    def __init__(self,p):
        self.p = p

    # Propabillity Density function
    def pdf(self,x,mu,kappa):
        Wp = self.c(self.p,kappa) * np.exp(kappa * (mu.T @ x )**2)
        return Wp

    # Log Likelihood
    def log_likelihood(self,mu,k,X):
        n = len(X[0])
        y = 0
        likelihood = n * (k * mu.T @ self.Scatter_matrix(X) @ mu - np.log(self.M(1/2,self.p/2,k)) + y)
        return likelihood
    
    # Estimation of Scatter Matrix
    def Scatter_matrix(self,X):
        S = np.zeros((self.p,self.p))
        for x in X.T:
            S += np.outer(x,x)
        return S
    
    # Gamme function defined for n >1
    def Gamma(self,n):
        return faculty(n-1)

    # Kummers geometric function
    def M(self,a,c,k):
        
        M0 = 1
        Madd = 1

        for j in range(1,100000):
            Madd = Madd * (a+j-1)/(c+j-1) * k/j
            M0 += Madd
            if Madd < 1e-10:
                break
        return M0

    # Normalization term for PDF
    def c(self,p,k):
        return self.Gamma(p/2) / (2 * np.pi**(p/2) * self.M(1/2,p/2,k))
