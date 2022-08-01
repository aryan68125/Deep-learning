import numpy as np
from process import get_data

#First get our data
X,Y = get_data()

#Now we are going to randomly initialize the weights of our neural network since we don't know how to train them yet
#Number of hidden layers
M = 5
#Number of inputs
D = X.shape[1]
#number of classes (K)--> number of unique values in Y
# so we will assume that that's numbered zero to k-1
K = len(set(Y))

#initialize the weights
#On the input side
#W1 weight matrix has to be D x M matrix
W1 = np.random.randn(D,M)
#b1 bias term
#size M vector
b1 = np.zeros(M)
#On the output side
#M2 weight matrix has to be M X K matrix
W2 = np.random.randn(M,K)
#b2 bias term
#size K vector
b2 = np.zeros(K)

#softmax function
def softmax(a):
    #performing softmax operation
    '''
    perform softmax operation on the activation
    step 1 : exponentiate these activation
    step 2 : divide by the sum of expa
    '''
    expA = np.exp(a) #Step 1
    Result = expA/expA.sum(axis=1,keepdims=True) #Step 2
    return Result

#function to move into forward direction
def forward(X,W1,b1,W2,b2):
    #to calculate the hidden layer values we are gonna use tanh activation function
    Z = np.tanh(X.dot(W1)+b1)

    #Now we can calculate the softmax of the next layer
    A = Z.dot(W2) +b2
    #perform softmax operation on A
    softmax_calculation = softmax(A)
    return softmax_calculation

#function to calculate the accuracy
def classification_rate(Y,P):
    #Boolean is treated as 0 and 1 so it's number correct divided by number total
    return np.mean(Y==P)

#Once we have done all this Now we can get the actual output
P_Y_given_X = forward(X,W1,b1,W2,b2)
predictions = np.argmax(P_Y_given_X,axis=1)
print(f"Classification rate Score : {classification_rate(Y,predictions)}")
