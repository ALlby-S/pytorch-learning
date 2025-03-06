#%% packages
import numpy as np
import pandas as pd
import torch
import torch.nn as nn 
import seaborn as sns

#%% data import
cars_file = 'https://gist.githubusercontent.com/noamross/e5d3e859aa0c794be10b/raw/b999fb4425b54c63cab088c0ce2c0d6ce961a563/cars.csv'
cars = pd.read_csv(cars_file)
cars.head()

#%% visualise the model
sns.scatterplot(x='wt', y='mpg', data=cars)
sns.regplot(x='wt', y='mpg', data=cars)

#%% convert data to tensor
x_list = cars.wt.values
#Converting list to numpy array of floats, then formatting them as a n by 1 matrix (many rows, 1 column)
x_np = np.array(x_list,dtype=np.float32).reshape(-1,1) #32 observations/data points, 1 column/feature
y_list = cars.mpg.values.tolist()

#convert data to tensor:
x = torch.from_numpy(x_np) #array to tensor, we can also go from
y = torch.tensor(y_list) #list to tensor


#%% training
weight = torch.rand(1,requires_grad=True,dtype=torch.float32) #give initial random value to weights
bias = torch.rand(1,requires_grad=True,dtype=torch.float32) #give random initial value to bias

num_epochs = 1000 #one epoch is one full set of training data passing through the neural network
learning_rate = 0.001

for epoch in range(num_epochs):
	for i in range(len(x)):
		# forward pass
		y_prediction = x[i] * weight + bias

		# calculate loss
		loss_tensor = torch.pow(y_prediction - y[i],2) #loss formula. remember y contains the "answers", aka dependent variable value.
		
		#backward pass
		loss_tensor.backward()

		# extract losses
		loss_value = loss_tensor.data[0]

		# update weights and biases
		with torch.no_grad():
			weight -= weight.grad * learning_rate
			bias -= bias.grad * learning_rate

			weight.grad.zero_() #underscore at end of method name indicate "in-place operation", does not return anything but changes values at execution.
			bias.grad.zero_()
	print(loss_value)
#%% check results
print(f"Weight: {weight.item()}, Bias:{bias.item()}")
# Note that weight represents the gradient, bias represents the Y intercept, see graph:
sns.scatterplot(x='wt', y='mpg', data=cars)
sns.regplot(x='wt', y='mpg', data=cars)


# %% Can take the predictions out an convert to numpy array:
y_predictions = ((x * weight) + bias).detach().numpy()

sns.scatterplot(x=x_list,y=y_list)
sns.lineplot(x=x_list, y=y_predictions.reshape(-1)) #got error that data must be 1D, it was 2D, use reshape(-1) to remove a dimension.
# %% (Statistical) Using a "real" Linear Regression model to compare results to the Neural Network we just made above.
from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(x_np,y_list) #fitting independent variables data points to our target results
# extract the gradient and intercept:
print(f"regression model slope:{reg.coef_} and Intercept:{reg.intercept_}")
print("Values found by our own model were:")
print(f"Weight: {weight.item()}, Bias:{bias.item()}")
# %% create graph visualisation
# make sure GraphViz is installed (https://graphviz.org/download/)
# if not computer restarted, append directly to PATH variable
# import os
# from torchviz import make_dot
# os.environ['PATH'] += os.pathsep + 'C:/Program Files (x86)/Graphviz/bin'
# make_dot(loss_tensor)
# %%
