import torch
from torchviz import make_dot

v = torch.tensor(1.0, requires_grad=True)
print(type(make_dot(v)))