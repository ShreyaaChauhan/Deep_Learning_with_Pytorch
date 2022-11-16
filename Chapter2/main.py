from __future__ import annotations

import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from helper_functions import make_train_step_fn
from helper_functions import make_val_step_fn
from helper_functions import mini_batch
from helper_functions import save_checkpoint
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.utils.data.dataset import random_split
from torch.utils.tensorboard import SummaryWriter

plt.style.use('fivethirtyeight')


device = 'cuda' if torch.cuda.is_available() else 'cpu'
parent_folder_path = os.path.abspath(os.path.dirname(__file__))

# Data Prepration
torch.manual_seed(13)
true_b = 1
true_w = 2
N = 100
x = np.random.rand(N, 1)
y = true_b + true_w * x + (.1 * np.random.randn(N, 1))

# Tensor from numpy arrays
x_tensor = torch.from_numpy(x).float().to(device)
y_tensor = torch.from_numpy(y).float()


# Builds dataset containing ALL data points
dataset = TensorDataset(x_tensor, y_tensor)

# Performs the split
ratio = .8
n_total = len(dataset)
n_train = int(n_total * ratio)
n_val = n_total - n_train

train_data, val_data = random_split(dataset, [n_train, n_val])

# DataLoader
train_loader = DataLoader(dataset=train_data, batch_size=16, shuffle=True)
val_loader = DataLoader(dataset=val_data, batch_size=16)

# set learning rate
lr = 0.1

torch.manual_seed(42)
# create or set your model here
model = nn.Sequential(nn.Linear(1, 1).to(device))
# optimizer
optimizer = optim.SGD(model.parameters(), lr=lr)

# MSE loss function
loss_fn = nn.MSELoss(reduction='mean')

# Create a train_step function for our model, loss dunction and optimizer
train_step = make_train_step_fn(model, loss_fn, optimizer)

# create a val_step function for our model and loss function
val_step = make_val_step_fn(model, loss_fn)

# Create the Summary Writer to interface with TensorBoard
writer = SummaryWriter(model, loss_fn)

# fetched a single mini -batch so we can use add_graph
x_dummy, y_dummy = next(iter(train_loader))
writer.add_graph(model, x_dummy.to(device))

n_epochs = 100

losses = []
val_losses = []

n_checkpoint_epoch = 50
try:
    shutil.rmtree(
        os.path.join(
            parent_folder_path, 'checkpoints',
        ),
    )
except Exception as e:
    print(e)

for epoch in range(n_epochs):
    ckpt_dir_path = os.path.join(
        os.path.abspath('.'), 'Chapter2', 'checkpoints',
    )
    os.makedirs(ckpt_dir_path, exist_ok=True)
    loss = mini_batch(device, train_loader, train_step)
    losses.append(loss)

    with torch.no_grad():
        val_loss = mini_batch(device, val_loader, val_step)
        val_losses.append(val_loss)

    writer.add_scalars(
        main_tag='loss', tag_scalar_dict={
            'training': loss, 'validation': val_loss,
        }, global_step=epoch,
    )

    if epoch % n_checkpoint_epoch == 0:

        checkpoint = {
            'epoch': n_epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': losses,
            'val_loss': val_losses,
        }
        save_checkpoint(epoch, checkpoint)
save_checkpoint(epoch, checkpoint, LATEST=True)

checkpoint = torch.load(
    os.path.join(
        parent_folder_path, 'checkpoints/latest.pth',
    ),
)

model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

saved_epoch = checkpoint['epoch']
saved_losses = checkpoint['loss']
saved_val_losses = checkpoint['val_loss']
epochs = 200
model.train()
# for epoch in range(saved_epoch, epochs + 1):
#     ckpt_dir_path = os.path.join(
#         os.path.abspath('.'), 'Chapter2', 'checkpoints')
#     os.makedirs(ckpt_dir_path, exist_ok=True)
#     loss = mini_batch(device, train_loader, train_step)
#     losses.append(loss)
#     with torch.no_grad():
#         val_loss = mini_batch(device, val_loader, val_step)
#         val_losses.append(val_loss)
#     writer.add_scalars(main_tag='loss', tag_scalar_dict={
#         'training': loss, 'validation': val_loss}, global_step=epoch)
#     if epoch % n_checkpoint_epoch == 0:

#         checkpoint = {
#             'epoch': n_epochs,
#             'model_state_dict': model.state_dict(),
#             'optimizer_state_dict': optimizer.state_dict(),
#             'loss': losses,
#             'val_loss': val_losses
#         }
#         save_checkpoint(epoch, checkpoint)
# save_checkpoint(epoch, checkpoint, LATEST=True)
# Close the writer
writer.close()
