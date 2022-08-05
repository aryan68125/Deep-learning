import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from process import get_data
'''
neural network for multiclass classification
'''

#We are gonna need a function to get the indicator matrix from the targets
# takes in Y output and K number of classes
def y2indicator(y,K):
    N = len(y)
    indicator_matrix = np.zeros((N,K))
    #So this is like a one hot encoding for the targets
    for i in range(N):
        indicator_matrix[i,y[i]] = 1
    return indicator_matrix

#Now we get our data
X,Y = get_data()
#Then shuffle our data
X,Y = shuffle(X,Y)
#convert Y to int32
Y = Y.astype(np.int32)

#defining the neural network architecture
#Number of inputs
D=X.shape[1]
#Number of hidden layers
M=5
#number of classes
K=len(set(Y))

#split our data into train and test sets
#train dataset
Xtrain = X[:-100]
Ytrain = Y[:-100]
Ytrain_indicator = y2indicator(Ytrain, K)
#test dataset
Xtest = X[-100:]
Ytest = Y[-100:]
Ytest_indicator = y2indicator(Ytest, K)

#Now we can randomly initialize our weights for the neural networks
W1 = np.random.randn(D,M)
b1 = np.zeros(M)
W2 = np.random.randn(M,K)
b2 = np.zeros(K)

#define the sofmax function
def softmax(a):
    expA = np.exp(a)
    #divide by sum along the
    summision = expA / expA.sum(axis=1,keepdims = True)
    return summision

#defining forward function(X,W1,b1,W2,b2):
#we have to return Z which is the value of the hidden units also along with the value of the softmax function calculation
def forward(X,W1,b1,W2,b2):
    # sigmoid
    # Z = 1 / (1 + np.exp( -(X.dot(W1) + b1) ))

    # tanh
    # Z = np.tanh(X.dot(W1) + b1)
    Z = np.tanh(X.dot(W1)+b1) #input->hidden layers

    # relu
    # Z = X.dot(W1) + b1
    # Z = Z * (Z > 0)

    #Z.dot(W2) + b2 --> activation
    return softmax(Z.dot(W2) + b2),Z #softmax hidden->output layer

#define the predict function
def predict(P_Y_given_X):
    return np.argmax(P_Y_given_X,axis=1)

#define the function for classification_rate
def classification_rate(Y,P):
    return np.mean(Y==P)

#function to calculate the cross_entropy for multi class classification
# T -> Targets
def cross_entropy(T,pY):
    return -np.mean(T * np.log(pY))

#training the neural network via loop
#list to store the train and test costs
train_costs = []
test_costs = []
learning_rate = 0.001
#loop for training the neural network
for i in range(99999):
    pYtrain, Ztrain = forward(Xtrain,W1,b1,W2,b2)
    pYtest, Ztest = forward(Xtest,W1,b1,W2,b2)

    #now we can calculate the cost
    ctrain = cross_entropy(Ytrain_indicator,pYtrain)
    ctest = cross_entropy(Ytest_indicator,pYtest)
    train_costs.append(ctrain)
    test_costs.append(ctest)

    #gradient descent for finding optimum weights for our neural network
    W2 -= learning_rate * Ztrain.T.dot(pYtrain - Ytrain_indicator) #[pYtrain = Output (Y)] - [Ytrain_indicator = Targets (T)]
    b2 -= learning_rate * (pYtrain - Ytrain_indicator).sum() #[pYtrain = Output (Y)] - [Ytrain_indicator = Targets (T)]
    #we need error at hidden nodes so that would be
    # derivative of tanh = (1-Ztrain*Ztrain)
    '''
    [pYtrain = Output (Y)] - [Ytrain_indicator = Targets (T)]
    dz = (Y-T).dot(W2.T) * Z * (1-Z) --> sigmoid activation function
    dZ = (Y - T).dot(W2.T) * (1-Ztrain*Ztrain) -->tanh activation function
    dZ = (Y - T).dot(W2.T) * (Z > 0) --> relu activation function
    delta_J_wrt_W = X.T.dot(dz)
    '''
    dZ = (pYtrain - Ytrain_indicator).dot(W2.T) * (1-Ztrain*Ztrain) # activation function
    W1 -= learning_rate * Xtrain.T.dot(dZ)
    b1 -= learning_rate * dZ.sum(axis=0)

    if i % 1000 == 0:
        print(f" i = {i} ; ctrain = {ctrain} ; ctest = {ctest}")

#finally we can print the classification rate and print the cost
print(f"Final train classification rate : {classification_rate(Ytrain, predict(pYtrain))}")
print(f"Final test classification rate : {classification_rate(Ytest, predict(pYtest))}")

#plot the cost
legend1, = plt.plot(train_costs)
legend2, = plt.plot(test_costs)
plt.legend([legend1,legend2],["train_costs", "test_costs"])
plt.show()
