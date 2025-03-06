#%% packages
import numpy as np
# %%
X = [0, 1]
w1 = [2, 3]
w2 = [0.4, 1.8]

# %% Question: which weight is more similar to input data X
dot_X_w1 = X[0] * w1[0] + X[1] * w1[1]
dot_X_w1
#%% Using numpy
dot_X_w2= np.dot(X,w2)
dot_X_w2
#%% Which weight vector (w) corresponds closer to X?
print("dot product of X and w1 is: {}".format(dot_X_w1))
print("dot product of X and w2 is: {}".format(dot_X_w2))
print("The higher dot product indicates better alignment between vectors")

# %%
