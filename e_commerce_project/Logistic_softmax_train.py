import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from process import get_data
'''
Logistic regression multiclass classification with softmax
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
D = X.shape[1]
#K the number of classes assuming our classes are numbered from 0 to k-1
K = len(set(Y))

#split our data into train and test sets
#train dataset
Xtrain = X[:-100]
Ytrain = Y[:-100]
Ytrain_indicator = y2indicator(Ytrain, K)
#test dataset
Xtest = X[-100:]
Ytest = Y[-100:]
Ytest_indicator = y2indicator(Ytest, K)

#initialize our weights
W = np.random.randn(D, K)
b = np.zeros(K)


#let's define our softmax function
def softmax(a):
    expA = np.exp(a)
    #divide by sum along the
    summision = expA / expA.sum(axis=1,keepdims = True)
    return summision

#defining forward function(X,W,b):
def forward(X,W,b):
    return softmax(X.dot(W) + b)

def predict(P_Y_given_X):
    return np.argmax(P_Y_given_X,axis=1)

def classification_rate(Y,P):
    return np.mean(Y==P)

def cross_entropy(T,pY):
    return -np.mean(T*np.log(pY))

#training loop starts here
#here we are gonna keep track of train costs
train_costs = []
#here we are gonna keep track of test costs
test_costs = []
learning_rate = 0.001
for i in range(99999):
    pYtrain = forward(Xtrain, W, b)
    pYtest = forward(Xtest, W, b)

    ctrain = cross_entropy(Ytrain_indicator, pYtrain)
    ctest = cross_entropy(Ytest_indicator, pYtest)
    train_costs.append(ctrain)
    test_costs.append(ctest)

    #performing gradient descent
    W -= learning_rate * Xtrain.T.dot(pYtrain - Ytrain_indicator)
    b -= learning_rate * (pYtrain - Ytrain_indicator).sum(axis=0)
    if i % 1000 == 0:
        print(f"i : {i} ; ctrain : {ctrain} ; ctest : {ctest}")

print(f"Final train classification rate : {classification_rate(Ytrain, predict(pYtrain))}")
print(f"Final test classification rate : {classification_rate(Ytest, predict(pYtest))}")

#plot the cost
legend1, = plt.plot(train_costs)
legend2, = plt.plot(test_costs)
plt.legend([legend1,legend2],["train_costs", "test_costs"])
plt.show()
