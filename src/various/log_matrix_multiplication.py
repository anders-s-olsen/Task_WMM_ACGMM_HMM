import torch
import torch.nn as nn
import numpy as np


def log_matmul(A, B):
    # A and B should be log
    Astack = torch.permute(torch.stack([A] * B.shape[1]), (1, 0, 2))
    Bstack = torch.permute(torch.stack([B] * A.shape[0]), (0, 2, 1))
    log_multi = Astack + Bstack
    return torch.logsumexp(log_multi, dim=2)
