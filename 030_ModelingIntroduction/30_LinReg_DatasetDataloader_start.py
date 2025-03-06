#%% packages
import graphlib
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

#%% Create dataset and dataloader
class LinearRegressionDataSet(Dataset):
    def __init__(self,X,y): #Input the independent features, and dependent features.
        self.X = X
        self.y = y 

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index],self.y[index]
    
# Now we can load our data into our program through a DataLoader, which has extra functions
train_loader = DataLoader(dataset= LinearRegressionDataSet(X_np,y_np),batch_size=2)


#%%
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


#%% Check what the DataLoader can return:
for position, (independent,dependent) in enumerate(train_loader):
    print(f"{position}th batch")
    print(f"Independent values (Car Weight): {independent}\nDependent values (MPG):{dependent}")
# after printing, we can see that it returns 2 pairs of values, the first array contains the independent, 
# the second contains the dependent. Elements in each index corresponds from one array to the other, e.g. index = 0 corresponds to the same value in the second.
# Specifically it returns a list of Tensors
#%% perform training
losses = []
slope, bias = [], []
NUM_EPOCHS = 1000
BATCH_SIZE = 2 #Best practice as a baseline is to use 32.
for epoch in range(NUM_EPOCHS):
    for position,(X,y) in enumerate(train_loader):
        # optimization
        optimizer.zero_grad()

        # forward pass
        y_pred = model(X)

        # compute loss
        loss = loss_fun(y_pred, y) #By using the dataloader, we called train_loader, we can simply refer to the data point being used to train the model as X, rather than X[index], because the DataLoader takes care of providing the next data point.
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

    

# %% visualise model training
sns.scatterplot(x=range(len(losses)), y=losses)

#%% visualise the bias development
sns.lineplot(x=range(NUM_EPOCHS), y=bias)
#%% visualise the slope development
sns.lineplot(x=range(NUM_EPOCHS), y=slope)



# %% check the result
model.eval()
y_pred = [i[0] for i in model(X).data.numpy()]
y = [i[0] for i in y_true.data.numpy()]
sns.scatterplot(x=X_list, y=y)
sns.lineplot(x=X_list, y=y_pred, color='red')
# %%
import hiddenlayer as hl
graph = hl.build_graph(model, X)
# %%
