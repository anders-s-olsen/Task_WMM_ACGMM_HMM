import torch
import torch.nn as nn
import numpy as np
#from scipy.special import gamma, factorial

class Watson(nn.Module):
    """
    Logarithmic Multivariate Watson distribution class
    """

    def __init__(self, p,D=None):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.p = torch.tensor(p,device=self.device)
        self.c = torch.tensor(self.p/2,device=self.device)
        self.mu = nn.Parameter(nn.functional.normalize(torch.rand(self.p), dim=0).to(self.device))
        self.kappa = nn.Parameter(torch.randint(1,10,(1,),dtype=torch.float32).to(self.device))
        self.SoftPlus = nn.Softplus(beta=20, threshold=1)
        self.const_a = torch.tensor(0.5,device=self.device)  # a = 1/2,  !constant
        self.logSA = torch.lgamma(self.c) - torch.log(torch.tensor(2,device=self.device)) -self.c* torch.log(torch.tensor(np.pi,device=self.device))
        assert self.p != 1, 'Not properly implemented'

    def get_params(self):
        mu_param = nn.functional.normalize(self.mu.data, dim=0)
        kappa_param = self.kappa.data
        return {'mu': mu_param,
                'kappa': kappa_param}

    def log_sum(self,x,y):
        s = x + torch.log(1+torch.exp(y-x));
        return s

    #def log_kummer_anders(self,a,b,kappa):
    #    # Not implemented here in vector space
    #    kappa = kappa.to('cpu')
    #    max_iter = 5000
    #    tol = 10**(-10)
    #    foo = torch.zeros(1)
    #    logkum = torch.zeros(1)
    #    logkum_old = torch.ones(1)
    #    j = 1
    #    while torch.abs(logkum-logkum_old)>tol and j<max_iter:
    #        logkum_old = logkum
    #        foo = foo + torch.log((a+j-1)/(j*(b+j-1))*kappa)
    #        logkum = self.log_sum(logkum,foo)
    #        j+=1
    #    #print(j)
    #    return logkum


    def log_kummer(self, a, b, kappa,num_eval=10000):

        n = torch.arange(num_eval,device=self.device)
    
        inner = torch.lgamma(a + n) + torch.lgamma(b) - torch.lgamma(a) - torch.lgamma(b + n) \
                + n * torch.log(kappa) - torch.lgamma(n + torch.tensor(1.,device=self.device))
    
        logkum = torch.logsumexp(inner, dim=0)
    
        return logkum

    def log_sphere_surface(self):
        logSA = torch.lgamma(self.c) - torch.log(torch.tensor(2)) - self.c*torch.log(torch.tensor(np.pi))

        return logSA

    def log_norm_constant(self, kappa_pos):
        logC = self.logSA - self.log_kummer(self.const_a, self.c, kappa_pos)
        return logC

    def log_pdf(self, X):
        # Constraints
        kappa_positive = torch.clamp(self.SoftPlus(self.kappa), min=1e-25)  # Log softplus?

        mu_unit = nn.functional.normalize(self.mu, dim=0)  ##### Sufficent for backprop?

        if torch.isnan(torch.log(kappa_positive)):
            raise ValueError('Too low kappa')
            print('Code reached here') # if kappa is zeros
            kappa_positive += 1e-15

        norm_constant = self.log_norm_constant(kappa_positive)
        logpdf = norm_constant + kappa_positive * ((mu_unit @ X.T) ** 2)

        if torch.isnan(logpdf.sum()):
            print(kappa_positive)
            print(torch.log(kappa_positive))

            raise ValueError('NNAAANANANA')
        return logpdf

    def forward(self, X):
        return self.log_pdf(X)

    def __repr__(self):
        return 'Watson'


if __name__ == "__main__":
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import scipy
    mpl.use('Qt5Agg')

    W = Watson(p=2)
    print(W.log_kummer(torch.tensor(0.5),torch.tensor(100),torch.tensor(2)))
    print(W.log_kummer(torch.tensor(0.5),torch.tensor(100),torch.tensor(200)))
    print(W.log_kummer(torch.tensor(0.5),torch.tensor(100),torch.tensor(2000)))

    W_pdf = lambda phi: float(torch.exp(W(torch.tensor([np.cos(phi), np.sin(phi)], dtype=torch.float))))
    w_result = scipy.integrate.quad(W_pdf, 0., 2*np.pi)

    # phi = linspace(0, 2 * pi, 320);
    # x = [cos(phi);sin(phi)];
    #
    # _, inner = W.log_kummer(torch.tensor(0.5), torch.tensor(3/2), torch.tensor())

    phi = torch.arange(0, 2 * np.pi, 0.001)
    phi_arr = np.array(phi)
    x = torch.column_stack((torch.cos(phi), torch.sin(phi)))

    points = torch.exp(W(x))
    props = np.array(points.squeeze().detach())

    # props = props/np.max(props)

    ax = plt.axes(projection='3d')
    ax.plot(np.cos(phi_arr), np.sin(phi_arr), 0, 'gray')
    ax.scatter(np.cos(phi_arr), np.sin(phi_arr), props, c=props, cmap='viridis', linewidth=0.5)

    ax.view_init(30, 135)
    plt.show()
