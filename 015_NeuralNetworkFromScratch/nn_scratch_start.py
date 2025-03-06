
#%% packages
import numpy as np #Math library
import pandas as pd #Statistics library
import seaborn as sns #Graphs library
from sklearn.preprocessing import StandardScaler #Shortfor Science Kit Learn (open source machine learning library for Python)
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix



#%% data prep
# source: https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset
df = pd.read_csv('heart.csv')
df.head()



#%% separate independent / dependent features
# Notice below, we are taking all columns of the data except 'output'
x = np.array(df.loc[ :, df.columns != 'output']) #Independent variables, converted into numpy array

# We are saving only the 'output' column to the y array.
y = np.array(df['output']) #Dependent feature, converted into numpy array as well.

print(f"X: {x.shape}, y: {y.shape}") #Printing the dimensions of each, notice that they match in row count.


#%% Train / Test Split
# Below we feed our two arrays to the traint test split function, and indicate how much of our data want to to test with,
# used 20% (0.2). We have also applied a random seed, if we were to run this line again, we would get the same shuffle patter.s
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=123)



#%% scale the data
# Looking at the values contained in our data, we see a wide range, and different ranges for each independent variable.

# Starting by creatings a scaler object, it will scale data according to normal distribution.
scaler = StandardScaler()

# Apply the transform that will fit our training data to the normal distribution. This may also save the scaling factors used to fit our data,
# back into the scaler object we are using.
x_train_scale = scaler.fit_transform(x_train) #scaled TRAINING data

# We can use the scaling factors from the first transform (above) to the rest of our independent variables data (testing)
x_test_scale = scaler.transform(x_test) #scaled TESTING data

#%% network class
class NeuralNetworkFromScratch:
    def __init__(self, learning_rate, X_train,Y_train, X_test, Y_test):
        self.weights = np.random.randn(X_train.shape[1]) #initialise with random weights for all 13 independent variables
        self.bias = np.random.randn() #Assign a single float value
        
        self.learning_rate = learning_rate
        
        self.X_train = X_train
        self.Y_train = Y_train

        self.X_test = X_test  
        self.Y_test = Y_test

        self.training_losses = []
        self.testing_losses = []

    def activation(self,x):
        # Sigmoid function
        return 1/(1+np.exp(-x)) #standard function, looks like an S
    
    def deactivation(self,x):
        # Returns the derivative of the Sigmoid function
        return self.activation(x) * (1-self.activation(x))
    
    # Forward Pass
    def forward(self,X):
        hidden_1 = np.dot(X, self.weights) + self.bias #hidden state, (representation of previous inputs)
        activate_1 = self.activation(hidden_1)
        return activate_1
    

    def backward(self, X, y_true):
        # Calculating gradients
        hidden_1 = np.dot(X, self.weights) + self.bias #hidden state, (representation of previous inputs)
        y_predicted = self.forward(X) #Predicted Dependent variable from the Independent data we provided.
        dloss_dpred = 2 * (y_predicted - y_true) #Can find formula in pdf.  Derivative of lost with respect to Derivative of Hidden
        dpredictions_dhiddenlayer = self.deactivation(hidden_1) #deriv. of prediction with respect to deriv of Hidden
        dhidden_dbias = 1 #because bias is a variable, becomes 1.
        dhidden_dweight = X

        dloss_dbias = dloss_dpred * dpredictions_dhiddenlayer * dhidden_dbias #Help find values for updated Bias
        dloss_dweight = dloss_dpred * dpredictions_dhiddenlayer * dhidden_dweight #Help find values for updated Weights
        return dloss_dbias, dloss_dweight
    
    def optimiser(self, dloss_dbias, dloss_dweights):
        # Update weights based on input derivatives
        self.bias = self.bias - (dloss_dbias * self.learning_rate)
        self.weights = self.weights - (dloss_dweights * self.learning_rate)

    
    def train(self,iterations):
        for i in range(iterations):
            # set up a random position in the range of the training data's length
            random_pos = np.random.randint(len(self.X_train))

            # forward pass
            y_train_true = self.Y_train[random_pos] #The correct label/s for the data point at the random position.
            y_train_pred = self.forward(self.X_train[random_pos]) #The predicted label/s for the same random data point.

            # calculate the loss
            loss = np.sum(np.square(y_train_pred - y_train_true)) #formula can be found in the NNfromScratch pdf
            self.training_losses.append(loss) #For graphing later, save the loss value for each iteration

            # calculate gradients at the same random data point
            dloss_dbias, dloss_dweights = self.backward(self.X_train[random_pos], self.Y_train[random_pos])

            # update weights
            self.optimiser(dloss_dbias=dloss_dbias,dloss_dweights=dloss_dweights)

            # calculate error for test data
            loss_sum = 0
            for j in range(len(self.X_test)):
                y_true = self.Y_test[j]
                y_pred = self.forward(self.X_test[j])
                loss_sum+= np.square(y_pred - y_true)
            self.testing_losses.append(loss_sum)
        return "Training successful!!!!!!!"
    


#%% Hyper parameters
LEARNING_RATE = 0.1
ITERATIONS = 1000



#%% model instance and training
# Instance the NN:
neural_network = NeuralNetworkFromScratch(learning_rate=LEARNING_RATE,X_train=x_train_scale,Y_train=y_train, X_test=x_test_scale,Y_test=y_test)
# Train for n iterations:
neural_network.train(iterations=ITERATIONS)



# %% check losses
# Plot the training losses on a simple line graph
sns.lineplot(x = list(range(len(neural_network.testing_losses))), y = neural_network.testing_losses)



# %% iterate over test data, test the NN with new data
total = x_test_scale.shape[0]
correct = 0
y_predictions = []

for i in range(total):
    y_true = y_test[i] #grab correct label for data point.
    y_predicted = np.round(neural_network.forward(x_test_scale[i])) #obtain NN predicted value for data same data point.
    y_predictions.append(y_predicted) #save the predicted value to an array.

    correct += 1 if y_true == y_predicted else 0 #add to the total if NN guessed correctly



# %% Calculate Accuracy
accuracy = correct / total * 100
print("Got an accuracy of: {}%".format(round(accuracy,2)))


# %% Baseline Classifier
from collections import Counter
Counter(y_test) #Counts how many of each classification exist in our data, running this for the example library indicates: {1: 31, 0: 30}, about a 50/50 split.
# Baseline accuracy would be 31/60 (about 50%, since our NN performed above this, it has improved.)


# %% Confusion Matrix
confusion_matrix(y_true=y_test, y_pred=y_predictions) #prints an array, diagonal values in '\' indicate correct guesses (25 + 250).
# off-diagonal values in '/' indicate incorreect guesses (6 + 7)
# Confusion matrix is a standard graph, plotting True positives, true negatives, false positives and false negatives.



# %% Conclusion
# Now that the model has been evaluated, we can start tuning.
# Tunable parameters include the learning rate