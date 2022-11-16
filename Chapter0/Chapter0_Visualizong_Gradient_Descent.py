from __future__ import annotations

import numpy as np

# Model y = b + wx + e
true_b = 1
true_w = 2
N = 100

# Data Generation
# It guarantees that every time we
# run this code, same random numbers will be generated
np.random.seed(42)
x = np.random.rand(N, 1)
epsilon = (.1 * np.random.randn(N, 1))
y = true_b + true_w + epsilon

# Shuffles the indices
idx = np.arange(N)
np.random.shuffle(idx)

# Uses first 80 random indices for train
train_idx = idx[:int(N*.8)]
val_idx = idx[int(N*8):]

# Generates train and test validation sets
x_train, y_train = x[train_idx], y[train_idx]
x_val, y_val = x[val_idx], y[val_idx]


# Step 0 - Random Initialization

np.random.seed(42)
b = np.random.randn(1)
w = np.random.randn(1)

# Step 1 - Compute model's prediction
yhat = b + w*x_train
error = yhat - y_train
loss = (error**2).mean()

b_grad = 2*error.mean()
w_grad = 2 * (x_train*error).mean()
print(loss)
lr = 0.1
b = b-lr*b_grad
w = w-lr*w_grad
