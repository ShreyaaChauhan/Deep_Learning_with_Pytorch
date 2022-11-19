from __future__ import annotations

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from engine_4 import StepbyStep
from helper_function import generate_dataset
from helper_function import index_splitter
from helper_function import plot_images
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import SubsetRandomSampler
from torchvision.transforms import Compose
from torchvision.transforms import Normalize
from torchvision.transforms import RandomHorizontalFlip

# from helper_function import make_balanced_sampler
plt.style.use('fivethirtyeight')


images, labels = generate_dataset(
    img_size=5, n_images=300, binary=True, seed=13,
)

plot_images(images, labels, n_plot=30)

# data prepration
x_tensor = torch.as_tensor(images/225).float()
print(labels.shape)
y_tensor = torch.as_tensor(labels.reshape(-1, 1)).float()


class TransformedTensorDataset(Dataset):
    def __init__(self, x, y, transform=None):
        self.x = x
        self.y = y
        self.transform = transform

    def __getitem__(self, index):
        x = self.x[index]

        if self.transform:
            x = self.transform(x)

        return x, self.y[index]

    def __len__(self):
        return len(self.x)


composer = Compose([
    RandomHorizontalFlip(
        p=0.5,
    ), Normalize(mean=(.5), std=(.5)),
])
dataset = TransformedTensorDataset(x_tensor, y_tensor, composer)
train_idx, val_idx = index_splitter(len(x_tensor), [80, 20])
train_sampler = SubsetRandomSampler(train_idx)
val_sampler = SubsetRandomSampler(val_idx)
train_loader = DataLoader(
    dataset=dataset, batch_size=16, sampler=train_sampler,
)
val_loader = DataLoader(dataset=dataset, batch_size=16, sampler=val_sampler)
# x_train_tensor = x_tensor[train_idx]
# y_train_tensor = y_tensor[train_idx]
# x_val_tensor = x_tensor[val_idx]
# y_val_tensor = y_tensor[val_idx]
# train_composer = Compose(
#     [RandomHorizontalFlip(p=0.5), Normalize(mean=(.5), std=(.5))])
# val_composer = Compose([Normalize(mean=(.5), std=(.5))])
# train_dataset = TransformedTensorDataset(
#     x_train_tensor, y_train_tensor, transform=train_composer)
# val_dataset = TransformedTensorDataset(
#     x_val_tensor, y_val_tensor, transform=val_composer)
# train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
# val_loader = DataLoader(dataset=val_dataset, batch_size=16)
# sampler = make_balanced_sampler(y_train_tensor)
# train_loader = DataLoader(dataset=train_dataset,
#                           batch_size=16, sampler=sampler)
# val_loader = DataLoader(dataset=val_dataset, batch_size=16)
"""
# Sets learning rate - this is "eta" ~ the "n" like Greek letter
lr = 0.1

torch.manual_seed(17)
# Now we can create a model
model_logistic = nn.Sequential()
model_logistic.add_module('flatten', nn.Flatten())
model_logistic.add_module('output', nn.Linear(25, 1, bias=False))
model_logistic.add_module('sigmoid', nn.Sigmoid())

# Defines a SGD optimizer to update the parameters
optimizer_logistic = optim.SGD(model_logistic.parameters(), lr=lr)

# Defines a binary cross entropy loss function
binary_loss_fn = nn.BCELoss()

n_epochs = 100
ckpt_interval = 10
sbs_logistic = StepbyStep(
    model_logistic, binary_loss_fn,
    optimizer_logistic, ckpt_interval,
)
sbs_logistic.set_loaders(train_loader, val_loader)
sbs_logistic.train(n_epochs)
sbs_logistic.plot_losses()"""


lr = 0.1
ckpt_interval = 10
torch.manual_seed(17)
model_nn = nn.Sequential()
model_nn.add_module('flatten', nn.Flatten())
model_nn.add_module('hidden0', nn.Linear(25, 5, bias=False))
model_nn.add_module('hidden1', nn.Linear(5, 3, bias=False))
model_nn.add_module('output', nn.Linear(3, 1, bias=False))
model_nn.add_module('sigmoid', nn.Sigmoid())
# Defines a SGD optimizer to update the parameters
optimizer_logistic = optim.SGD(model_nn.parameters(), lr=lr)

# Defines a binary cross entropy loss function
binary_loss_fn = nn.BCELoss()

n_epochs = 100
ckpt_interval = 10
sbs_logistic = StepbyStep(
    model_nn, binary_loss_fn,
    optimizer_logistic, ckpt_interval,
)
sbs_logistic.set_loaders(train_loader, val_loader)
sbs_logistic.train(n_epochs)
sbs_logistic.plot_losses()
print(sbs_logistic.count_parameters())
print(model_nn[1].weight.data.numpy().shape)
