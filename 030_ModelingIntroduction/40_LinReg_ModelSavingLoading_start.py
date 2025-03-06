#%% packages
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader 
import seaborn as sns

#%% data import
cars_file = 'https://gist.githubusercontent.com/noamross/e5d3e859aa0c794be10b/raw/b999fb4425b54c63cab088c0ce2c0d6ce961a563/cars.csv'
cars = pd.read_csv(cars_file)
cars.head()

#%% visualise the model
sns.scatterplot(x='wt', y='mpg', data=cars)
sns.regplot(x='wt', y='mpg', data=cars)

#%% convert data to tensor
X_list = cars.wt.values
X_np = np.array(X_list, dtype=np.float32).reshape(-1,1)
y_list = cars.mpg.values
y_np = np.array(y_list, dtype=np.float32).reshape(-1,1)
X = torch.from_numpy(X_np)
y_true = torch.from_numpy(y_np)

#%% Dataset and Dataloader
class LinearRegressionDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(dataset = LinearRegressionDataset(X_np, y_np), batch_size=2) # Best practice is to use a batch size of 32 as a baseline


#%% Model
class LinearRegressionTorch(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegressionTorch, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        return self.linear(x)

input_dim = 1
output_dim = 1
model = LinearRegressionTorch(input_size=input_dim, output_size=output_dim)
model.train()

# %% Mean Squared Error
loss_fun = nn.MSELoss()

#%% Optimizer
learning_rate = 0.02
# test different values of too large 0.1 and too small 0.001
# best 0.02
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

#%% perform training
losses = []
slope, bias = [], []
number_epochs = 1000
for epoch in range(number_epochs):
    for j, data in enumerate(train_loader):
        # optimization
        optimizer.zero_grad()

        # forward pass
        y_hat = model(data[0])

        # compute loss
        loss = loss_fun(y_hat, data[1])
        losses.append(loss.item())

        # backprop
        loss.backward()

        # update weights
        optimizer.step()
    
    # get parameters
    for name, param in model.named_parameters():
        if param.requires_grad:
            if name == 'linear.weight':
                slope.append(param.data.numpy()[0][0])
            if name == 'linear.bias':
                bias.append(param.data.numpy()[0])


    # store loss
    losses.append(float(loss.data))
    # print loss
    if (epoch % 100 == 0):
        print(f"Epoch {epoch}, Loss: {loss.data}")

# %% model state dict
model.state_dict() #returns the linear weights and biases tensors


# %% save model state dict
torch.save(model.state_dict(),'model_state_dict.pth') 
# Pass the model's state dictionary to save, and the filepath to save it to.
# Above we have saved the model to this file path, with the name in the quotes with the .pth extension
# It is possible to save the entire model, but this save format is subject to code breaking if it is used in a different file path
# Therefore, it is better to save only the weights and biases, which is done by using the state_dict() method of the model being saved.


# %% load a model
model_2 = LinearRegressionTorch(input_size=input_dim, output_size=output_dim) #WHen loding a model we actually create a new instance of the LinRegTorch class we made above, and use the saved state dictionary from the saved model.
model_2.load_state_dict(torch.load('model_state_dict.pth')) #Loading the state dictionary from prev model to the new instance of the LinRegTorch model.

# We can verify that the model was loaded correctly by checking the output of the following line, matches the values of the original model:
model_2.state_dict()

# %%
