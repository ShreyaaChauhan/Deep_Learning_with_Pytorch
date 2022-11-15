from engine import StepbyStep
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


# data deneration
true_b = 1
true_w = 2
N = 100

# Data Generation
np.random.seed(42)
x = np.random.rand(N, 1)
y = true_b + true_w * x + (.1 * np.random.randn(N, 1))

# Shuffles the indices
idx = np.arange(N)
np.random.shuffle(idx)

# Uses first 80 random indices for train
train_idx = idx[:int(N*.8)]
# Uses the remaining indices for validation
val_idx = idx[int(N*.8):]

# Generates train and validation sets
x_train, y_train = x[train_idx], y[train_idx]
x_val, y_val = x[val_idx], y[val_idx]

# Data prepration
torch.manual_seed(13)
x_tensor = torch.as_tensor(x).float()
y_tensor = torch.as_tensor(y).float()
dataset = TensorDataset(x_tensor, y_tensor)
ratio = .8
n_total = len(dataset)
n_train = int(n_total * ratio)
n_val = n_total - n_train
train_data, val_data = random_split(dataset, [n_train, n_val])
train_loader = DataLoader(dataset=train_data, batch_size=16, shuffle=True)
val_loader = DataLoader(dataset=val_data, batch_size=16)

# model configurations
lr = 0.1
ckpt_interval = 10
torch.manual_seed(42)
model = nn.Sequential(nn.Linear(1, 1))
optimizer = optim.SGD(model.parameters(), lr=lr)
loss_fn = nn.MSELoss(reduction='mean')

sbs = StepbyStep(model, loss_fn, optimizer, ckpt_interval)
sbs.set_loaders(train_loader, val_loader)
sbs.set_tensorboard('classy')
sbs.train(n_epochs=20)
print(model.state_dict())
print(sbs.total_epochs)

# new_data = np.array([.5, .6, .7]).reshape(-1, 1)
# predictions = sbs.predict(new_data)
# print(predictions)
# new_data = np.array([.5]).reshape(-1, 1)
# predictions = sbs.predict(new_data)
# print(model.state_dict())


sbs.load_checkpoint(
    '/Users/shreyachauhan/Deep_Learning_with_Pytorch/Chapter2.1/checkpoints/latest.pth')
sbs.set_loaders(train_loader, val_loader)
sbs.train(n_epochs=80)
print(model.state_dict())
