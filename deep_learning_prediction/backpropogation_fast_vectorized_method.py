import numpy as np
import matplotlib.pyplot as plt

#let's define the forward action of the neural network
#It's gonna take in x matrix , W1, b1, W2, b2
def forward(X,W1,b1,W2,b2):
    # we are gonna use sigmoid non linearity in the hidden layers
    #Z is the value at the hidden layer
    # sigmoid(-(a+b1)) --> sigmoid(-a-b1) hence exp(-X.dot(W1)-b1)
    Z = 1/(1+np.exp(-X.dot(W1)-b1))

    #Now we can calculate the softmax of the next layer
    A = Z.dot(W2) +b2
    #Perform softmax operation on A --> softmax(A)
    #we exponentiate A
    expA = np.exp(A)
    #Output Y
    Y = expA/expA.sum(axis=1,keepdims = True)
    #Here we are return Y and Z both because it is required to calculate the gradient descent
    return Y,Z

#define a function to calculate the classification rate
#This is gonna take in targets Y and predictions P
def classification_rate(Y,P):
    n_correct = 0
    n_total = 0
    for i in range(len(Y)):
        n_total +=1
        if Y[i]==P[i]:
            n_correct +=1
    return float(n_correct)/n_total

'''############################################################define the derivative functions###################################################'''
#derivative function with respect to w2
#this function will take in the Z=hidden layer values , T = Targets and Y = Outputs
def derivative_w2(Z,T,Y):
    #get the shape N and K from the targets matrix
    N,K = T.shape
    #Get the number of hidden units in hidden layer from Z.shape[1]
    M = Z.shape[1]

    #slow way of doing this -> So this is directly from our derivation
    # ret1 = np.zeros((M,K))
    # for n in range(N):
    #     for m in  range(M):
    #         for k in range(K):
    #             #here we are doing T-Y because we are doing gradient ascent
    #             ret1[m,k] += (T[n,k] - Y[n,k])*Z[n,m]

    #fast way of doint this --> Vectorized numpy calculations
    #step 1 simplification get rid of the m
    # ret2 = np.zeros((M,K))
    # for n in range(N):
    #     for k in range(K):
    #         #here we are doing T-Y because we are doing gradient ascent
    #         #so there are 2 1D arrays on each side ret1 and Z
    #         ret2[:,k] += (T[n,k] - Y[n,k])*Z[n,:]
    #assert(np.abs(ret1 - ret2).sum() < 10e-10) --> sanity check

    #step 2 simplification get rid of the k
    # ret3 = np.zeros((M,K))
    # for n in range(N):
    #     #The all of ret3 will be the outer product of Zn , Tn - Yn
    #     ret3 += np.outer(Z[n],T[n] - Y[n])
    #assert(np.abs(ret2 - ret3).sum() < 10e-10) --> sanity check

    #step 3 simplification get rid of the loops completely
    ret4 = Z.T.dot(T-Y)
    return ret4

#derivative with respect to b2
#This function will take in T = Targets and Y = Outputs
def derivative_b2(T,Y):
    derivation_wrt_b2 = (T-Y).sum(axis=0)
    return derivation_wrt_b2

#derivative with respect to W1
#SO this function takes in  X = the input matrix , Z = hidden values , T = Targets , Y = Outputs , W2 = Output layer weights
def derivative_w1(X, Z, T, Y, W2):
    #Get N and D from X.shape
    N,D = X.shape
    #Get M,K from W2.shape
    M,K = W2.shape

    #Slow method
    # ret1 = np.zeros((D,M))
    # for n in range(N):
    #     for k in range(K):
    #         for m in range(M):
    #             for d in range(D):
    #                 ret1[d,m] += (T[n,k] - Y[n,k])*W2[m,k]*Z[n,m]*(1-Z[n,m])*X[n,d]

    #fast method --> numpy vectorized calulation method
    #delta at Z
    dz = (T-Y).dot(W2.T) * Z * (1-Z)
    delta_J_wrt_W = X.T.dot(dz)
    return delta_J_wrt_W

#derivative with respect to b1
#This takes in T = Targets , Y = Outputs , W2 = Weights Output layer weights , Z = Hidden layer values
def derivative_b1(T, Y, W2, Z):
     derivation_wrt_b1 = ((T-Y).dot(W2.T) * Z * (1-Z)).sum(axis=0)
     return derivation_wrt_b1
'''##############################################################################################################################################'''

#defining the cost function
#This takes in the targets = T and the outputs = Y
def cost(T, Y):
    tot = T * np.log(Y)
    J_cost_function = tot.sum()
    return J_cost_function

#main function is going to be consist of creating a data
def main():
    '''creating the data'''
    #let's create 500 samples per class
    #So what we are gonna do is generate some gaussian clouds
    Nclass = 500

    #Number of inputs
    D=2
    #Number of hidden layers
    M=3
    #number of classes
    K=3

    #So we are gonna have 3 gaussian clouds
    #np.random.randn(Nclass, 2) + np.array([0,-2]) So the 1st gaussian cloud is gonna centered at 0,-2
    X1 = np.random.randn(Nclass, D) + np.array([0,-2])
    #np.random.randn(Nclass, 2) + np.array([2,2]) So the 2nd gaussian cloud is gonna centered at 2,2
    X2 = np.random.randn(Nclass, D) + np.array([2,2])
    #np.random.randn(Nclass, 2) + np.array([-2,2]) So the 3rd gaussian cloud is gonna centered at -2,2
    X3 = np.random.randn(Nclass, D) + np.array([-2,2])
    X = np.vstack([X1,X2,X3])

    #create our labels
    Y = np.array([0]*Nclass + [1]*Nclass + [2]*Nclass)

    N = len(Y)
    #Turn the targets into an indicator variable because we expect those to be either zero or one
    #where as in Y variable we represent the classes by zero to k-1
    #We need an indicator variable of size N by K
    T = np.zeros((N,K))
    #So this is like a one hot encoding for the targets
    for i in range(N):
        T[i,Y[i]] = 1

    #visualizing the created data points on matplotlib
    plt.scatter(X[:,0], X[:,1], c=Y, s=100, alpha=0.5)
    plt.show()

    #Starting part of this is the same so we are going to randomly initialize the weights
    #randomly initialize the weights
    #On the input side
    #W1 weight matrix has to be D x M matrix
    W1 = np.random.randn(D,M)
    #b1 bias term
    b1 = np.random.randn(M)
    #On the output side
    #M2 weight matrix has to be M X K matrix
    W2 = np.random.randn(M,K)
    #b2 bias term
    b2 = np.random.randn(K)

    '''doing back propogation '''
    learning_rate = 10e-7
    #array of cost function so that we can plot it afterwards to see the progression
    costs= []
    #now I am gonna do 500000 epochs
    for epoch in range(999999):
        #This version of the forward function not only returns the output but also returns the hidden layer
        output,hidden = forward(X,W1,b1,W2,b2)
        #every 100 epochs we are gonna calculate the cost and print it
        if (epoch % 100==0):
            #calculating cost
            c = cost(T,output)
            #calculating predictions
            P = np.argmax(output, axis = 1)
            #calculating the classification rate
            r = classification_rate(Y,P)
            print(f"cost : {c} , classification_rate : {r}")
            #append the cost to the cost array
            costs.append(c)

        #now we are going to do gradient ascent this is just the backwards of gradient descent
        W2 += learning_rate * derivative_w2(hidden, T, output)
        b2 += learning_rate * derivative_b2(T, output)
        W1 += learning_rate * derivative_w1(X, hidden, T, output, W2)
        b1 += learning_rate * derivative_b1(T, output, W2, hidden)

    #finally when all that's done we can plot the cost
    plt.plot(costs)
    plt.show()

if __name__ == '__main__':
    main()
