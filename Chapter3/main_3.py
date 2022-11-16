from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from engine_3 import StepbyStep
from helper_function_3 import precision_recall
from helper_function_3 import sigmoid
from helper_function_3 import split_cm
from helper_function_3 import tpr_fpr
from sklearn.datasets import make_moons
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

# data generation
X, y = make_moons(n_samples=100, noise=0.3, random_state=0)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=.2, random_state=13,
)

sc = StandardScaler()
sc.fit(X_train)

X_train = sc.transform(X_train)
X_val = sc.transform(X_val)

# Data prepration
torch.manual_seed(13)

# Builds tensors from numpy arrays
x_train_tensor = torch.as_tensor(X_train).float()
y_train_tensor = torch.as_tensor(y_train.reshape(-1, 1)).float()

x_val_tensor = torch.as_tensor(X_val).float()
y_val_tensor = torch.as_tensor(y_val.reshape(-1, 1)).float()

# Builds dataset containing ALL data points
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
val_dataset = TensorDataset(x_val_tensor, y_val_tensor)

# Builds a loader of each set
train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=16)


# model configuration
lr = 0.1
ckpt_interval = 10
torch.manual_seed(42)
model = nn.Sequential()
model.add_module('linear', nn.Linear(2, 1))
optimizer = optim.SGD(model.parameters(), lr=lr)
loss_fn = nn.BCEWithLogitsLoss()
n_epoch = 100

sbs = StepbyStep(model, loss_fn, optimizer, ckpt_interval)
sbs.set_loaders(train_loader, val_loader)
sbs.set_tensorboard('chapter3')
sbs.train(n_epochs=100)
sbs.plot_losses()


logit_val = sbs.predict(X_val)
probabilities_val = sigmoid(logit_val).squeeze()
cm_thresh50 = confusion_matrix(y_val, (probabilities_val >= 0.5))
TN, FP, FN, TP = split_cm(cm_thresh50)
print(tpr_fpr(cm_thresh50))

print(cm_thresh50)
print(model.state_dict())
print(precision_recall(cm_thresh50))
threshs = np.linspace(0, 1, 11)
print(threshs)
fpr, tpr, threshold1 = roc_curve(y_val, probabilities_val)
prec, rec, threshold2 = precision_recall_curve(y_val, probabilities_val)
