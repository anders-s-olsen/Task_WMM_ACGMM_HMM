import torch
import torch.nn as nn

import numpy as np
import scipy
from Watson_torch import Watson
from AngularCentralGauss_torch import AngularCentralGaussian

torch.set_printoptions(precision=15)

W = Watson(p=2)
ACG = AngularCentralGaussian(p=2)


W_pdf = lambda phi: float(torch.exp(W(torch.tensor([np.cos(phi), np.sin(phi)], dtype=torch.float))))
ACG_pdf = lambda phi: float(torch.exp(ACG(torch.tensor([[np.cos(phi), np.sin(phi)]], dtype=torch.float))))


w_result = scipy.integrate.quad(W_pdf, 0., 2*np.pi)
acg_result = scipy.integrate.quad(ACG_pdf, 0., 2*np.pi)

print(f'Integral test of Watson and ACG pdf yield the following\n Watson: {w_result} \n ACG: {acg_result}')