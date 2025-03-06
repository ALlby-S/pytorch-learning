#%% packages
from ast import Mult
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader 
import seaborn as sns
import numpy as np
from collections import Counter
# %% data prep
X, y = make_multilabel_classification(n_samples=10000, n_features=10, n_classes=3, n_labels=2)
X_torch = torch.FloatTensor(X)
y_torch = torch.FloatTensor(y)

# %% train test split
X_train, X_test, y_train, y_test = train_test_split(X_torch, y_torch, test_size = 0.2)


# %% dataset and dataloader
class MultilabelDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# TODO: create instance of dataset
multilabel_train_data = MultilabelDataset(X_train,y_train)
multilabel_test_data = MultilabelDataset(X_test,y_test)
# TODO: create train loader
train_loader = DataLoader(dataset=multilabel_train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(dataset=multilabel_test_data, batch_size=32, shuffle=True)

# %% model
# TODO: set up model class
# topology: fc1, relu, fc2
# final activation function??
class MultiLabelNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MultiLabelNet,self).__init__()
        self.fc1 = nn.Linear(input_size,hidden_size) #fully connected layer
        self.fc2 = nn.Linear(hidden_size,output_size) #fully connected layer
        self.relu = nn.ReLU() #Rectified Linear Unit ReLU
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        x = self.fc1(x) #first input layer
        x = self.relu(x) #activation function for next layer?
        x = self.fc2(x) #second layer
        x = self.sigmoid(x) #activation function for output node/s
        return x

        


# TODO: define input and output dim
input_dim = multilabel_train_data.X.shape[1]
output_dim = len(multilabel_train_data.y.unique())

num_features = X.shape[1]
num_classes = y.shape[1]
hidden_features = 20
# TODO: create a model instance
model = MultiLabelNet(input_size=num_features,output_size=num_classes,hidden_size=hidden_features)

# %% loss function, optimizer, training loop
# TODO: set up loss function and optimizer
loss_fn = nn.BCEWithLogitsLoss() #Binary Cross Entroy, best suited for sigmoid activation function, outputs 1 or 0 only.
lr = 0.01
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
losses = []
slope, bias = [], []
number_epochs = 100

# TODO: implement training loop
for epoch in range(number_epochs):
    for j, (X,y) in enumerate(train_loader):
        
        # optimization
        optimizer.zero_grad()

        # forward pass
        y_pred = model(X)

        # compute loss
        loss = loss_fn(y_pred,y)
        
        # backward pass
        loss.backward()

        # update weights
        optimizer.step()
        
    # TODO: print epoch and loss at end of every 10th epoch
    if epoch % 10 == 0:
        print(f'epoch: {epoch}, loss: {loss.data.item()}')
        losses.append(loss.item())
    
# %% losses
# TODO: plot losses
sns.scatterplot(x=range(len(losses)), y=losses)
# %% test the model
# TODO: predict on test set
with torch.no_grad(): #testing model without updating its parameters
    y_test_pred = model(X_test).round() #feed test data and round the results


#%% Naive classifier accuracy
# TODO: convert y_test tensor [1, 1, 0] to list of strings '[1. 1. 0.]'
# requires list comprehension
# Currently, y_test is a tensor of rounded floats, we want to convert these to array of strings for each value point
y_test_string = [str(i) for i in y_test.detach().numpy()]

# TODO: get most common class count
most_common_result = Counter(y_test_string).most_common()[0][1]

# TODO: print naive classifier accuracy
print(f"Naive Classifer Accuracy: {most_common_result/len(y_test) * 100}%")
# Above returns 21.9% therefore if we just guessed the most common result we would only
# have an accuracy of 21.9%


# %% Test accuracy
# TODO: get test set accuracy
accuracy_score(y_test,y_test_pred)
# This model is 68.8% accurate with the test data, therefore we can say our model is
# successful.
# %%



