import numpy as np
import torch
import torch.nn as nn

from scipy.special import gamma

#device = 'cpu'
class AngularCentralGaussian(nn.Module):
    """
    Angular-Central-Gaussian spherical distribution:

    "Tyler 1987 - Statistical analysis for the angular central Gaussian distribution on the sphere"
    """

    def __init__(self, p):
        super().__init__()

        self.p = p
        # assert self.p % 2 == 0, 'P must be an even positive integer'
        self.half_p = torch.tensor(p / 2)
        self.L_diag = nn.Parameter(torch.rand(self.p))
        self.L_under_diag = nn.Parameter(torch.tril(torch.randn(self.p, self.p), -1))
        self.SoftPlus = nn.Softplus()
        assert self.p != 1, 'Not matmul not stable for this dimension'

    def get_params(self):
        return self.Alter_compose_A(read_A_param=True)

    def log_sphere_surface(self):
        logSA = torch.lgamma(self.half_p) - torch.log(2 * np.pi ** self.half_p)
        return logSA

    def Alter_compose_A(self, read_A_param=False):

        """ Cholesky Component -> A """
        L_diag_pos_definite = self.SoftPlus(self.L_diag)  # this is only semidefinite...Need definite
        L_inv = torch.tril(self.L_under_diag, -1) + torch.diag(L_diag_pos_definite)
        log_det_A = -2 * torch.sum(torch.log(L_diag_pos_definite))  # - det(A)

        if read_A_param:
            return torch.linalg.inv((L_inv.T @ L_inv))

        return log_det_A, L_inv

    def compose_A(self, read_A_param=False):

        L_diag_pos_definite = self.SoftPlus(self.L_diag)  #this is only semidefinite...Need definite

        L = torch.tril(self.L_under_diag, -1) + torch.diag(L_diag_pos_definite)
        A = L @ L.T

        log_det_A = 2 * torch.sum(torch.log(L_diag_pos_definite))

        if read_A_param:
            return A

        return log_det_A, A

    # Probability Density function
    def log_pdf(self, X):
        log_det_A, L_inv = self.Alter_compose_A()


        #matmul1 = torch.diag(X @ L_inv @ X.T)

        B = X @ L_inv
        matmul2 = torch.sum(B * B, dim=1)

        if torch.isnan(matmul2.sum()):
            print(matmul2)
            print(L_inv)
            print(self.L_diag)
            raise ValueError('NaN was produced!')

        log_acg_pdf = self.log_sphere_surface() \
                      - 0.5 * log_det_A \
                      - self.half_p * torch.log(matmul2)

        return log_acg_pdf

    def forward(self, X):
        return self.log_pdf(X)

    def __repr__(self):
        return 'ACG'


if __name__ == "__main__":
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.use('Qt5Agg')
    dim = 3
    ACG = AngularCentralGaussian(p=dim)

    X = torch.randn(6, dim)

    out = ACG(X)
    print(out)
    # # ACG.L_under_diag = nn.Parameter(torch.ones(2,2))
    # # ACG.L_diag = nn.Parameter(torch.tensor([21.,2.5]))
    # phi = torch.arange(0, 2*np.pi, 0.001)
    # phi_arr = np.array(phi)
    # x = torch.column_stack((torch.cos(phi),torch.sin(phi)))
    #
    # points = torch.exp(ACG(x))
    # props = np.array(points.squeeze().detach())
    #
    # ax = plt.axes(projection='3d')
    # ax.plot(np.cos(phi_arr), np.sin(phi_arr), 0, 'gray') # ground line reference
    # ax.scatter(np.cos(phi_arr), np.sin(phi_arr), props, c=props, cmap='viridis', linewidth=0.5)
    #
    # ax.view_init(30, 135)
    # plt.show()
    # plt.scatter(phi,props, s=3)
    # plt.show()
