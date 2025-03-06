#%% packages
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import seaborn as sns
# %% data import
iris = load_iris()
X = iris.data
y = iris.target

# %% train test split
#Reserving 20% of data for testing.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2) 

# %% Pre-prossesing: convert to float32
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# %% custom dataset class
class IrisData(Dataset):
	def __init__(self,X_train,y_train):
		super().__init__()
		self.X = torch.from_numpy(X_train)
		self.y = torch.from_numpy(y_train)
		self.y = self.y.type(torch.LongTensor)
		self.len = self.X.shape[0]

	def __getitem__(self, index):
		return self.X[index], self.y[index]
	
	def __len__(self):
		return self.len


# %% dataloader
iris_data = IrisData(X_train, y_train)
train_loader = DataLoader(iris_data, batch_size=32, shuffle=True)

# %% check dims
print(f"X shape: {iris_data.X.shape}, y shape: {iris_data.y.shape}")
#the above line prints:
# X shape: torch.Size([120, 4]), y shape: torch.Size([120])
#which tells us that we are using 120 items, with 4 inputs, and 1 output each


# %% define class
class MultiClassNet(nn.Module):
	def __init__(self, NUM_FEATURES, NUM_CLASSES, HIDDEN_FEATURES):
		super().__init__()
		# Linear layer 1 takes number of features as input, and outputs number of hidden features.
		self.lin1 = nn.Linear(NUM_FEATURES,HIDDEN_FEATURES)

		# Linear layer 2 takes number of hidden features as input (from lin1), and outputs the number of classes, 
		# in this case, the types of iris flowers from the data set.
		self.lin2 = nn.Linear(HIDDEN_FEATURES,NUM_CLASSES)

		# Log soft max activation function (for multiclass pridiction)
		self.log_softmax = nn.LogSoftmax(dim=1)

	def forward(self, x):
		x = self.lin1(x) #passing x into the first layer, saving the output
		x = torch.sigmoid(x) #passing x through sigmoid activation function
		x = self.lin2(x) #passing  activation value through layer 2
		x = self.log_softmax(x) #passing layer 2 outputs, to multiple 
								#classifications activation function
		return x #return the classifications identified
	
	
# %% hyper parameters
NUM_FEATURES = iris_data.X.shape[1] #features = independent variables being used, here we have 4
HIDDEN = 6
NUM_CLASSES = len(iris_data.y.unique()) #we use 0,1,2 to show which class is detected, 
										# therefore 3 different classes of iris
# %% create model instance
model = MultiClassNet(NUM_FEATURES=NUM_FEATURES, NUM_CLASSES=NUM_CLASSES, HIDDEN_FEATURES=HIDDEN)


# %% loss function
criterion = nn.CrossEntropyLoss() #Cross Entropy Loss pairs well with log softmax activation function


# %% optimiser
LR = 0.1
optimiser = torch.optim.SGD(model.parameters(),lr=LR)

# %% training
NUM_EPOCHS = 100
losses = []
for epoch in range(NUM_EPOCHS):
	for X, y in train_loader:
		# initialise gradients
		optimiser.zero_grad()

		# forward pass
		y_pred_logged = model(X)

		# calculate losses
		loss = criterion(y_pred_logged,y)

		# backward pass
		loss.backward()

		# update weights
		optimiser.step()

	losses.append(float(loss.data.detach().numpy()))


# %% show losses over epochs
sns.lineplot(x=range(NUM_EPOCHS), y=losses)
# lowest loss level = 0.30057
# most recent loss level = 0.39867


# %% test the model
X_test_torch = torch.from_numpy(X_test) #converting test data to torch tensor
# the no grad setting will disable adjusting model parameters,  we are just using the model, not training.
with torch.no_grad():
	y_test_log = model(X_test_torch)
	y_test_pred = torch.max(y_test_log.data,1) #checking answer for given data point, output 1D value.

# %% Accuracy
model_accuracy = accuracy_score(y_test,y_test_pred.indices.numpy()) #after first pass, accuracy = 0.9666666666666667

# %% Compare to naive classifier
# Naive classifier will always predict the most common class, good for benchmarking and validating model
from collections import Counter
Counter(y_test).most_common()
# the above returns the sorted array of tuples for most to least common values.
# returned: [(1, 11), (0, 10), (2, 9)]
# therefore class 1 appears 11 times in the test data

# getting the count for most frequent value:
most_common_count = Counter(y_test).most_common()[0][1]

# Display the accuracy of the naive classifier:
print(f"Naive Classifier accuracy: {most_common_count/len(y_test) * 100}%")
print(f"Trained model accuracy: {model_accuracy * 100}%")
# %%
