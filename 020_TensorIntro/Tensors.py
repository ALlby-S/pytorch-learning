#%% packages
import torch
import seaborn as sns
import numpy as np

#%% create a tensor
x = torch.tensor(5.5)

# %% simple calculations
y = x + 10
print(y)

# %% automatic gradient calculation
print(x.requires_grad)  # check if requires_grad is true, false if not directly specified

x.requires_grad_() # set requires grad to true, default True

#%% or set the flag directly during creation
x = torch.tensor(2.0, requires_grad=True)
print(x.requires_grad)
#%% function for showing automatic gradient calculation
def y_function(val):
    return (val-3) * (val-6) * (val-4)

x_range = np.linspace(0, 10, 101) #linearly spacing values for a range
x_range
y_range = [y_function(i) for i in x_range] #list comprehension. For each element i in x_range, feed it into the y_function and save to the list (y_range)
sns.lineplot(x = x_range, y = y_range) #plot our graph over the x_range we made above.

# %% define y as function of x
y = (x-3) * (x-6) * (x-4)
print(y)
# %%

# %% x -> y
# create a tensor with gradients enabled
x = torch.tensor(1.0, requires_grad=True)
# create second tensor depending on first tensor
y = (x-3) * (x-6) * (x-4)
# calculate gradients
y.backward()
# show gradient of first tensor
print(x.grad) #prints the gradient at x = 1.0
# %%Another example: x -> y -> z
x = torch.tensor(1.0, requires_grad=True)
y = x**3
z = 5*y - 4

# %%
z.backward()
print(x.grad)  # should be equal 5
# %% more complex network, 2 inputs (x11, x21)
#First (inputs)
x11 = torch.tensor(2.0, requires_grad=True)
x21 = torch.tensor(3.0, requires_grad=True)

#Equivalent to hidden layer
x12 = 5 * x11 - 3 * x21
x22 = 2 * x11**2 + 2 * x21

#output layer
y = 4 * x12 + 3 * x22

#Generate the gradients for each input node
y.backward()

print(x11.grad)
print(x21.grad)
# %%
