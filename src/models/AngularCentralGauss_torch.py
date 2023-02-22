import numpy as np
import torch
import torch.nn as nn

#from scipy.special import gamma

#device = 'cpu'
class AngularCentralGaussian(nn.Module):
    """
    Angular-Central-Gaussian spherical distribution:

    "Tyler 1987 - Statistical analysis for the angular central Gaussian distribution on the sphere"
    """

    def __init__(self, p,regu=0):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.p = p
        self.regu = regu
        self.num_params = int(self.p*(self.p-1)/2+self.p)
        # assert self.p % 2 == 0, 'P must be an even positive integer'
        self.half_p = torch.tensor(p / 2)
        self.logSA = torch.lgamma(self.half_p) - torch.log(torch.tensor(2)) -self.half_p* torch.log(torch.tensor(np.pi))
        #self.L_diag = nn.Parameter(torch.rand(self.p))
        #self.L_under_diag = nn.Parameter(torch.tril(torch.randn(self.p, self.p), -1))

        # Anders addition here: 
        #self.L_tri_inv = nn.Parameter(torch.tril(torch.randn(self.p, self.p)).to(self.device)) # addition
        self.L_vec = nn.Parameter(torch.randn(self.num_params).to(self.device)) # addition
        
        self.tril_indices = torch.tril_indices(self.p,self.p)

        self.diag_indices = torch.zeros(self.p).type(torch.LongTensor)
        for i in range(1,self.p+1):   
            self.diag_indices[i-1] = ((i**2+i)/2)-1


        self.SoftPlus = nn.Softplus()
        assert self.p != 1, 'Not matmul not stable for this dimension'

    def get_params(self):
        return self.Alter_compose_A(read_A_param=True)

    #def log_sphere_surface(self):
    #    #logSA = torch.lgamma(self.half_p) - torch.log(2 * np.pi ** self.half_p)
    #    logSA = torch.lgamma(self.half_p) - torch.log(torch.tensor(2)) -self.half_p* torch.log(torch.tensor(np.pi))
    #    return logSA

    def Alter_compose_A(self, read_A_param=False):

        """ Cholesky Component -> A """
        #L_diag_pos_definite = self.SoftPlus(self.L_diag)  # this is only semidefinite...Need definite
        #L_inv = torch.tril(self.L_under_diag, -1) + torch.diag(L_diag_pos_definite)
        
        L_tri_inv = torch.zeros(self.p,self.p).to(self.device)
        L_tri_inv[self.tril_indices[0],self.tril_indices[1]] = self.L_vec

        # addition with regularization
        if self.regu>0:
            
            A_inv = L_tri_inv@L_tri_inv.T
            fac = torch.sqrt(torch.linalg.matrix_norm(A_inv)**2/self.p**2)

            L_tri_inv = torch.linalg.cholesky(A_inv/fac+self.regu*torch.eye(self.p))

            #L_tri_inv = torch.linalg.cholesky(torch.matrix_exp(A_inv-A_inv.T)+self.regu*torch.eye(self.p))
            #L_tri_inv = torch.linalg.cholesky(torch.exp(-torch.trace(A_inv)/self.p)*torch.matrix_exp(A_inv)+self.regu*torch.eye(self.p))

            log_det_A_inv = 2 * torch.sum(torch.log(torch.abs(torch.diag(L_tri_inv))))  # - det(A)
            
            #print(log_det_A_inv)
        else:
            log_det_A_inv = 2 * torch.sum(torch.log(torch.abs(self.L_vec[self.diag_indices])))  # - det(A)
        
        
        if read_A_param:
            #return torch.linalg.inv((self.L_tri_inv @ self.L_tri_inv.T))
            return L_tri_inv

        return log_det_A_inv, L_tri_inv

    # Probability Density function
    def log_pdf(self, X):
        log_det_A_inv, L_tri_inv = self.Alter_compose_A()


        #matmul1 = torch.diag(X @ L_inv @ X.T)

        B = X @ L_tri_inv
        matmul2 = torch.sum(B * B, dim=1)

        if torch.isnan(matmul2.sum()):
            print(matmul2)
            print(L_tri_inv)
            print(self.L_diag)
            raise ValueError('NaN was produced!')
        
        log_acg_pdf = self.logSA + 0.5 * log_det_A_inv - self.half_p * torch.log(matmul2)

        return log_acg_pdf

    def forward(self, X):
        return self.log_pdf(X)
    
    def __repr__(self):
        return 'ACG'


if __name__ == "__main__":
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import scipy

    mpl.use('Qt5Agg')
    dim = 3
    ACG = AngularCentralGaussian(p=dim)
    
    #ACG_pdf = lambda phi: float(torch.exp(ACG(torch.tensor([[np.cos(phi), np.sin(phi)]], dtype=torch.float))))
    #acg_result = scipy.integrate.quad(ACG_pdf, 0., 2*np.pi)

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
