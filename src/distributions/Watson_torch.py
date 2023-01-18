import torch
import torch.nn as nn
import numpy as np
from scipy.special import gamma, factorial
device = 'cpu'
class Watson(nn.Module):
    """
    Logarithmic Multivariate Watson distribution class
    """

    def __init__(self, p):
        super().__init__()

        self.p = p
        self.mu = nn.Parameter(nn.functional.normalize(torch.rand(self.p), dim=0))
        self.kappa = nn.Parameter(torch.randint(1,10,(1,),dtype=torch.float32))
        self.SoftPlus = nn.Softplus(beta=20, threshold=1)
        self.const_a = torch.tensor(0.5)  # a = 1/2,  !constant
        assert self.p != 1, 'Not proper implemted'

    def get_params(self):
        mu_param = nn.functional.normalize(self.mu.data, dim=0)
        kappa_param = self.kappa.data
        return {'mu': mu_param,
                'kappa': kappa_param}

    def log_kummer(self, a, b, kappa):
        n = torch.arange(1000).to(device)

        inner = torch.lgamma(a + n) + torch.lgamma(b) - torch.lgamma(a) - torch.lgamma(b + n) \
                + n * torch.log(kappa) - torch.lgamma(n + torch.tensor(1.).to(device))

        logkum = torch.logsumexp(inner, dim=0).to(device)

        return logkum

    def log_sphere_surface(self):
        logSA = torch.lgamma(torch.tensor(self.p / 2)) - torch.log(2 * np.pi ** (torch.tensor(self.p / 2)))

        return logSA

    def log_norm_constant(self, kappa_pos):
        logC = self.log_sphere_surface() - self.log_kummer(self.const_a, torch.tensor(self.p / 2), kappa_pos)
        return logC

    def log_pdf(self, X):
        # Constraints
        kappa_positive = torch.clamp(self.SoftPlus(self.kappa), min=1e-25)  # Log softplus?

        mu_unit = nn.functional.normalize(self.mu, dim=0)  ##### Sufficent for backprop?

        if torch.isnan(torch.log(kappa_positive)):
            print('Code reached here') # if kappa is zeros
            kappa_positive += 1e-15


        logpdf = self.log_norm_constant(kappa_positive) + kappa_positive * (mu_unit @ X.T) ** 2

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

    mpl.use('Qt5Agg')

    W = Watson(p=2)
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
