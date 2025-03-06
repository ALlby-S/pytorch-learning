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
X_list = cars.wt.values
X_np = np.array(X_list, dtype=np.float32).reshape(-1,1)
y_list = cars.mpg.values
y_np = np.array(y_list, dtype=np.float32).reshape(-1,1)
X = torch.from_numpy(X_np)
y_true = torch.from_numpy(y_np)


#%% model class
class LinearRegressionTorch(nn.Module):
	def __init__(self,input_size,output_size):
		super(LinearRegressionTorch,self).__init__()
		# Set up the linear layer, inheriting from nn module
		self.linear = nn.Linear(input_size, output_size) #Pass the Nodes coming in, and Nodes coming out of the model

	def forward(self,x):
		out = self.linear(x)
		return out

# Setting up our model: 
input_dimension = 1
output_dimension = 1
model = LinearRegressionTorch(input_dimension,output_dimension) #instantiating a model object with parameters based on data dimensions


# %% Creating Loss function
loss_function = nn.MSELoss() #Mean square error loss function (same as before)


# %% Creating Optimiser
learning_rate = 0.06 #Hyper parameter
# Below we set up an optimiser object, we access the torch optimisers module, and use stochastic gradient descent, feed it our model's parameters and learning rate.
optimiser = torch.optim.SGD(model.parameters(), lr=learning_rate) #We are using optim module, specifically using the stochastic gradient descent (SGD)


# %% Train model
losses, slope, bias = [], [], [] #set up each array as empty

number_of_epochs = 1000 #Hyper parameter

for epoch in range(number_of_epochs):
	# set gradients to zero
	optimiser.zero_grad()

	# forward pass
	y_predicted = model(X) #feeding a data point from our independent variable data (vehicle weight, wt)

	# By this point we have our prediction and the true value for the given data point, we can calculate the loss
	loss = loss_function(y_predicted,y_true) #pass prediction and true value to the loss function we create above.
	
	# Update gradients
	loss.backward()

	# update weights
	optimiser.step()

	# get parameters
	for name,param in model.named_parameters(): #loop through the named parameters of our model
		if param.requires_grad: #only look through parameters that require gradient ascent
			if name == 'linear.weight': #only looking at the weight (weight of hidden layer, not weight of the cars) parameter
				slope.append(param.data.numpy()[0][0]) #saving the first object of the weight parameter for each data point to our list of weights
			if name == 'linear.bias':
				bias.append(param.data.numpy()[0]) #saving the first object of the bias parameter to the end of our list of biases

	# store losses
	losses.append(float(loss.data))

	# We can also print the loss every 100 epochs
	if epoch % 100 == 0:
		print('Epoch: {}, Loss: {:0.4f}'.format(epoch,loss.data))

# ########### First time running the code above, we noticed that losses were not decreasing. we missed a line of code:
# ########### Inserted in line 69 loss.backward(), this updates the gradients of the tensors with current and previous guess information


# %% Visualise the model training losses
sns.scatterplot(x=range(number_of_epochs),y=losses)

# %% Visualise the model training biases at each epoch
sns.scatterplot(x=range(number_of_epochs),y=bias)


# %% Can also graph the slope (weight of hidden layer)
sns.scatterplot(x=range(number_of_epochs),y=slope)

# %% Plotting the prediction
y_predictions = model(X).data.numpy().reshape(-1) #access output data from model after providing input data.
sns.scatterplot(x=X_list, y=y_list) #plotting correct/true values for each input data point.
sns.lineplot(x=X_list, y=y_predictions,color='red') #plotting the predicted value from our model for each data point.


# %% After some tuning, we end with: Bias = 37.2814, Slope = -5.343401
# We can compare this to the statistical method: 
from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(X_np,y_list) #fitting independent variables data points to our target results
# extract the gradient and intercept:
print(f"regression model slope:{float(reg.coef_)} and Intercept:{reg.intercept_}")
print("Values found by our own model were:")
print(f"Weight: {slope[-1]}, Bias:{bias[-1]}")


# %%
