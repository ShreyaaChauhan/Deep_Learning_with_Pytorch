from __future__ import annotations

import os

import numpy as np
import torch


def make_train_step_fn(model, loss_fn, optimizer):
    # Builds function that performs a step in the train loop
    def perform_train_step_fn(x, y):
        # Sets model to TRAIN mode
        model.train()

        # Step 1 - Computes our model's predicted output - forward pass
        yhat = model(x)
        # Step 2 - Computes the loss
        loss = loss_fn(yhat, y)
        # Step 3 - Computes gradients for both "a" and "b" parameters
        loss.backward()
        # Step 4 - Updates parameters using gradients and the learning rate
        optimizer.step()
        optimizer.zero_grad()

        # Returns the loss
        return loss.item()

    # Returns the function that will be called inside the train loop
    return perform_train_step_fn


def make_val_step_fn(model, loss_fn):
    # Builds function that performs a step in the validation loop
    def perform_val_step_fn(x, y):
        # Sets model to EVAL mode
        model.eval()

        # Step 1 - Computes our model's predicted output - forward pass
        yhat = model(x)
        # Step 2 - Computes the loss
        loss = loss_fn(yhat, y)
        # There is no need to compute Steps 3
        # and 4, since we don't update parameters during evaluation
        return loss.item()

    return perform_val_step_fn


def mini_batch(device, data_loader, step_fn):
    mini_batch_losses = []
    for x_batch, y_batch in data_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        mini_batch_loss = step_fn(x_batch, y_batch)
        mini_batch_losses.append(mini_batch_loss)

    loss = np.mean(mini_batch_losses)
    return loss


def save_checkpoint(epoch: int, checkpoint: dict, LATEST: bool = False):
    ckpt_dir_path = os.path.join(
        os.path.abspath(os.path.dirname(__file__)), 'checkpoints',
    )
    os.makedirs(ckpt_dir_path, exist_ok=True)
    checkpoint = checkpoint
    checkpoint_name = f'epoch_{epoch}.pth' if not LATEST else 'latest.pth'
    checkpoint_name = os.path.join(ckpt_dir_path, checkpoint_name)
    torch.save(checkpoint, checkpoint_name)
    print('checkpoint saved', checkpoint_name)
