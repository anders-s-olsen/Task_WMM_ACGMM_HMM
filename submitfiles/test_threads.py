import torch
print(torch.get_num_threads())

torch.set_num_threads(16)
print(torch.get_num_threads())