import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

from util import getData, softmax, cost2, y2indicator, error_rate,relu

class ANN(object):
    def __init__(self, M):
        self.M = M

    def fit(self, X, Y, Xvalid, Yvalid, learning_rate=1e-6, reg=1e-6, epochs=10000, show_fig=False):

        N, D = X.shape
        K = len(set(Y))
        T = y2indicator(Y)
        self.W1 = np.random.randn(D, self.M) / np.sqrt(D)
        self.b1 = np.zeros(self.M)
        self.W2 = np.random.randn(self.M, K) / np.sqrt(self.M)
        self.b2 = np.zeros(K)

        costs = []
        best_validation_error = 1
        for i in range(epochs):
            # forward propagation and cost calculation
            pY, Z = self.forward(X)

            # gradient descent step
            pY_T = pY - T
            self.W2 -= learning_rate*(Z.T.dot(pY_T) + reg*self.W2)
            self.b2 -= learning_rate*(pY_T.sum(axis=0) + reg*self.b2)
            # dZ = pY_T.dot(self.W2.T) * (Z > 0) # relu
            dZ = pY_T.dot(self.W2.T) * (1 - Z*Z) # tanh
            self.W1 -= learning_rate*(X.T.dot(dZ) + reg*self.W1)
            self.b1 -= learning_rate*(dZ.sum(axis=0) + reg*self.b1)

            if i % 10 == 0:
                pYvalid, _ = self.forward(Xvalid)
                c = cost2(Yvalid, pYvalid)
                costs.append(c)
                e = error_rate(Yvalid, np.argmax(pYvalid, axis=1))
                print("i:", i, "cost:", c, "error:", e)
                if e < best_validation_error:
                    best_validation_error = e
        print("best_validation_error:", best_validation_error)

        if show_fig:
            plt.plot(costs)
            plt.show()


    def forward(self, X):
        # Z = relu(X.dot(self.W1) + self.b1)
        Z = np.tanh(X.dot(self.W1) + self.b1)
        return softmax(Z.dot(self.W2) + self.b2), Z

    def predict(self, X):
        pY, _ = self.forward(X)
        return np.argmax(pY, axis=1)

    def score(self, X, Y):
        prediction = self.predict(X)
        return 1 - error_rate(Y, prediction)

def main():
    X, Y = getData()
    #shuffle X and Y
    X,Y = shuffle(X, Y)
    #split X and y in taining and validation sets so we are going to use another set of data to plot the cost
    Xvalid,Yvalid = X[-1000:],Y[-1000:]
    Tvalid = y2indicator(Yvalid)
    # Set the X and Y to the rest of X and Y
    X, Y = X[:-1000],Y[:-1000]
    # #precedure to increase the number of one samples because there is a class imbalance
    # X0 = X[Y==0, :]
    # X1 = X[Y==1, :]
    # #repeate the occourances of data of X1
    # X1 = np.repeat(X1, 9, axis=0)
    # X = np.vstack([X0,X1])
    # Y = np.array([0]*len(X0) + [1]*len(X1))

    #use our model the way we use scikitLearn
    #100 is the size of the hidden layers
    model = ANN(100)
    model.fit(X,Y,Xvalid,Yvalid,show_fig = True)
    print(f"Final score : {model.score(X,Y)}")

if __name__=='__main__':
    main()
