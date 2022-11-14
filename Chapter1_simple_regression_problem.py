
from sklearn.linear_model import LinearRegression
from torchviz import make_dot
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from chapter1_figures import figure1, figure3
plt.style.use('fivethirtyeight')


device = 'cuda' if torch.cuda.is_available() else 'cpu'
true_b = 1
true_w = 2
N = 100

# Data Generation
np.random.seed(42)
x = np.random.rand(N, 1)
epsilon = (.1 * np.random.randn(N, 1))
y = true_b + true_w * x + epsilon
# Shuffles the indices
idx = np.arange(N)
np.random.shuffle(idx)

# Uses first 80 random indices for train
train_idx = idx[:int(N*.8)]
# Uses the remaining indices for validation
val_idx = idx[int(N*.8):]

# Generates train and validation sets
x_train, y_train = torch.as_tensor(x[train_idx]).float().to(
    device), torch.as_tensor(y[train_idx]).float().to(device)
x_val, y_val = torch.as_tensor(x[val_idx]).float().to(
    device), torch.as_tensor(y[val_idx]).float().to(device)


figure1(x_train, y_train, x_val, y_val)

# step 0 Initialization of parameter b and w randomly
# recommended way we can specify the device at the momet of creation
torch.manual_seed(42)


lr = 0.1


class ManualLinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.b = nn.Parameter(torch.randn(
            1, requires_grad=True, dtype=torch.float))
        self.w = nn.Parameter(torch.randn(1,
                                          requires_grad=True,
                                          dtype=torch.float))

    def forward(self, x):
        return self.b + self.w * x


#model = ManualLinearRegression().to(device)
model = nn.Sequential(nn.Linear(1, 5), nn.Linear(5, 1)).to(device)
# Defines a SGD optimizer to update the parameters
optimizer = optim.SGD(model.parameters(), lr=lr)
# Defines a MSE loss function
loss_fn = nn.MSELoss(reduction='mean')
n_epochs = 340
for epoch in range(n_epochs):
    # Step 1 Compute Model's prediction. This is a forward pass
    model.train()
    yhat = model(x_train)
    loss = loss_fn(yhat, y_train)
    loss.backward()
    print(loss.item(), epoch)
    # step 3 .
    optimizer.step()
    optimizer.zero_grad()

print(model.state_dict())
figure3(x_train, y_train)
