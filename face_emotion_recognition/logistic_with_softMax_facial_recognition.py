import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from util import getData, softmax, cost, error_rate, y2indicator
'''
facial expression recognition problem using logistic regression
code the Logisitc regression model in form of a class so we can use the object of the model
similar to the sciKitLearn library
'''
class LogisticModel(object):
    def __init__(self):
        pass
    #fit function which will train our model
    def fit(self, X,Y,learning_rate = 10e-8, reg=10e-12,epochs=10000, show_fig=False):
        #shuffle X and Y
        X,Y = shuffle(X, Y)
        #split X and y in taining and validation sets so we are going to use another set of data to plot the cost
        Xvalid,Yvalid = X[-1000:],Y[-1000:]
        Tvalid = y2indicator(Yvalid)
        # Set the X and Y to the rest of X and Y
        X, Y = X[:-1000],Y[:-1000]
        N,D = X.shape
        #Number of classes
        K = len(set(Y))
        #matrix of targets
        T = y2indicator(Y)
        #initialize the weights
        self.W = np.random.randn(D,K)/np.sqrt(D + K)
        self.b = np.zeros(K)

        # create an array to hold that cost
        costs = []
        # keep track of best validation
        # Its gonna start as 1
        best_validation_error = 1
        for i in range(epochs):
            #calculate probability Y given x
            pY = self.forward(X)

            #gradient descent step
            self.W -= learning_rate*(X.T.dot(pY-T) + reg*self.W)
            self.b -= learning_rate*((pY-T).sum(axis=0) + reg*self.b)

            #in every 20 steps we are going to calculate the cost
            if i%10 == 0:
                pYvalid = self.forward(Xvalid)
                #calculating cost
                c = cost(Tvalid,pYvalid)
                costs.append(c)
                #calculating error rate
                e = error_rate(Yvalid,np.argmax(pYvalid, axis = 1))
                print (f"i: {i} , cost : {c} , error : {e}")
                #keep track of best validation error
                if e < best_validation_error:
                    best_validation_error = e
        print(f"best validation error : {best_validation_error}")

        if show_fig:
            plt.plot(costs)
            plt.show()
    def forward(self, X):
        return softmax(X.dot(self.W) + self.b)
    def predict(self, X):
        pY = self.forward(X)
        return np.argmax(pY, axis = 1)
    #This could be used for cross validation
    def score(self,X,Y):
        prediction = self.predict(X)
        return 1- error_rate(Y,prediction)

def main():
    #get binary data This does not automatically balances the classes of data for us
    X,Y = getData()
    # # We are going to balance the classes of data here
    # X0 = X[Y==0, :]
    # X1 = X[Y==1, :]
    # #repeate the occourances of data of X1
    # X1 = np.repeat(X1, 9, axis=0)
    # X = np.vstack([X0,X1])
    # Y = np.array([0]*len(X0) + [1]*len(X1))

    #use our model the way we use scikitLearn
    model = LogisticModel()
    model.fit(X,Y,epochs=10000,show_fig = True)
    model.score(X,Y)

if __name__ == '__main__' :
    main()
