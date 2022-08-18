import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

from util import getBinaryData, sigmoid, sigmoid_cost, error_rate, relu

class ANN(object):
    #in the init here we are going to pass the hidden unit of this model
    def __init__(self, M):
        self.M = M

    def fit(self, X, Y, learning_rate = 5*10e-7, reg = 1.0, epochs = 10000, show_fig = False):
        X, Y = shuffle(X,Y)
        Xvalidation, Yvalidation = X[-1000:], Y[-1000:]
        X, Y = X[:-1000], Y[:-1000]
        N,D = X.shape

        #initialize the weights
        self.W1 = np.random.randn(D,self.M) / np.sqrt(D + self.M) #hidden weights
        self.b1 = np.zeros(self.M)
        #hidden to output weight
        self.W2 = np.random.randn(self.M) / np.sqrt(self.M)
        self.b2 = 0

        costs=[]
        best_validation_error = 1
        for i in range(epochs):
            #forward propogation
            pY, Z = self.forward(X)

            #gradient descent
            pY_Y = pY - Y
            self.W2 -= learning_rate*(Z.T.dot(pY_Y) + reg*self.W2)
            self.b2 -= learning_rate*((pY_Y).sum() + reg*self.b2)
            #backpropogation on the input to hidden layers
            # dZ = np.outer(pY_Y, self.W2) * (Z>0)
            dZ = np.outer(pY_Y, self.W2) * (1-Z * Z)
            self.W1 -= learning_rate*(X.T.dot(dZ) + reg*self.W1)
            self.b1 -= learning_rate*(np.sum(dZ, axis=0) + reg*self.b1)

            if i%10==0:
                pYvalid, _ = self.forward(Xvalidation)
                c = sigmoid_cost(Yvalidation, pYvalid)
                costs.append(c)
                e = error_rate(Yvalidation, np.round(pYvalid))
                print(f"epochs : {i} ; costs : {c} ; error rate : {e}")
                if e<best_validation_error:
                    best_validation_error=e
        print(f"best validation error : {best_validation_error}")

        if show_fig:
            plt.plot(costs)
            plt.show()

    def forward(self, X):
         # Z = relu(X.dot(self.W1) + self.b1)
         Z = np.tanh(X.dot(self.W1) + self.b1)
         output = sigmoid(Z.dot(self.W2) + self.b2)
         return output, Z

    def predict(self, X):
        pY, _ = self.forward(X)
        return np.round(pY)

    def score(self, X, Y):
        prediction = self.predict(X)
        return 1-error_rate(Y,prediction)

def main():
    X, Y = getBinaryData()

    #precedure to increase the number of one samples because there is a class imbalance
    X0 = X[Y==0, :]
    X1 = X[Y==1, :]
    #repeate the occourances of data of X1
    X1 = np.repeat(X1, 9, axis=0)
    X = np.vstack([X0,X1])
    Y = np.array([0]*len(X0) + [1]*len(X1))

    #use our model the way we use scikitLearn
    #100 is the size of the hidden layers
    model = ANN(100)
    model.fit(X,Y,show_fig = True)

if __name__=='__main__':
    main()
