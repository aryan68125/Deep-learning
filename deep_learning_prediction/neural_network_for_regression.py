from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# NOTE: some people using the default Python
# installation on Mac have had trouble with Axes3D
# Switching to Python 3 (brew install python3) or
# using Linux are both viable work-arounds

# generate and plot the data
'''
1. Here we are creating 500 uniformly spaced points between -2 and +2 on a 2D grid.
2. Then we multiply the first column of Xs which represents the first feature by the second column of Xs which represents the second
   feature to get Y
3. Next we do a 3D scatter plot of the data to see what it looks like
'''
N = 500
X = np.random.random((N, 2))*4 - 2 # in between (-2, +2)
Y = X[:,0]*X[:,1] # makes a saddle shape
# note: in this script "Y" will be the target,
#       "Yhat" will be prediction

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0], X[:,1], Y)
plt.show()

# make a neural network and train it
'''
1. We define D which is the dimensionality of the inputs
2. Choose the number of hidden units. Here I've chosen it to be 100
'''
#Number of inputs
D = 2
# number of hidden units
M = 100
'''
Randomly initialize the weights for the hidden layer and the output layer
'''
# layer 1 Hidden layer
W = np.random.randn(D, M) / np.sqrt(D)
b = np.zeros(M)

# layer 2 Output layer
V = np.random.randn(M) / np.sqrt(M)
c = 0

'''
Next we have a function to calculate the output
So remember we are doing regression and not a classification so we don't need a softmax function here at the end
And don't forget to return Z which is an intermediate value because it's used in the gradient descent calulation
'''
# how to get the output
# consider the params global
def forward(X):
  Z = X.dot(W) + b
  Z = Z * (Z > 0) # relu activation
  # Z = np.tanh(Z) for tanh activation
  # Z = 1 / (1 + np.exp( -Z )) for sigmoid activation
  # Z = Z * (Z > 0) for relu activation

  Yhat = Z.dot(V) + c
  return Z, Yhat



'''
Next we have some functions to calculate the derivatives and do the gradient descent update
Note that Y is the target and Yhat is the output
'''
# how to train the params
def derivative_V(Z, Y, Yhat):
  return (Y - Yhat).dot(Z)

def derivative_c(Y, Yhat):
  return (Y - Yhat).sum()

def derivative_W(X, Z, Y, Yhat, V):
  # dZ = np.outer(T-Y, W2) * Z * (1 - Z) # this is for sigmoid activation
  # dZ = np.outer(T-Y, W2) * (1 - Z * Z) # this is for tanh activation
  # dZ = np.outer(T-Y, W2) * (Z > 0) # this is for relu activation
  # T(Targets) = Y , Y(Output) = Yhat, W2 = V
  dZ = np.outer(Y - Yhat, V) * (Z > 0) # relu
  return X.T.dot(dZ)

def derivative_b(Z, Y, Yhat, V):
  # dZ = np.outer(T-Y, W2) * Z * (1 - Z) # this is for sigmoid activation
  # dZ = np.outer(T-Y, W2) * (1 - Z * Z) # this is for tanh activation
  # dZ = np.outer(T-Y, W2) * (Z > 0) # this is for relu activation
  dZ = np.outer(Y - Yhat, V) * (Z > 0) # this is for relu activation
  return dZ.sum(axis=0)

#updating weights for the neural network by performing gradient ascent
def update(X, Z, Y, Yhat, W, b, V, c, learning_rate=1e-4):
  gV = derivative_V(Z, Y, Yhat)
  gc = derivative_c(Y, Yhat)
  gW = derivative_W(X, Z, Y, Yhat, V)
  gb = derivative_b(Z, Y, Yhat, V)

  V += learning_rate*gV
  c += learning_rate*gc
  W += learning_rate*gW
  b += learning_rate*gb

  return W, b, V, c



'''
Function to calcualte the cost
The cost function used here is a mean squared error ---> ((Y - Yhat)**2).mean()
'''
# so we can plot the costs later
def get_cost(Y, Yhat):
  return ((Y - Yhat)**2).mean()



# run a training loop
# plot the costs
# and plot the final result
costs = []
for i in range(200):
  Z, Yhat = forward(X)
  W, b, V, c = update(X, Z, Y, Yhat, W, b, V, c)
  cost = get_cost(Y, Yhat)
  costs.append(cost)
  if i % 25 == 0:
    print(cost)

# plot the costs
plt.plot(costs)
plt.show()

'''
We want to plot the function that our neural network has learned and compare it against the data
1. So to do that we first repeat the code that we saw before to make the scatterplot
2. And after that I am going to create 20 evenly spaced points on the X1,X2 plane in both directions. To do that I need a function called mesh grid.
   Which basically gives me back every X1 X2 coordinate in that box. But the short version is we need to flatten those arrays and then stack them
   beside each other to get an NxD matrix. That I can finally pass in the neural newtork
3. SO I do just that and I get a new Yhat. Then I can use this plot_trisurf function to make a surface plot for all my new Yhats
'''
# plot the prediction with the data
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0], X[:,1], Y)

# surface plot
line = np.linspace(-2, 2, 20)
xx, yy = np.meshgrid(line, line)
Xgrid = np.vstack((xx.flatten(), yy.flatten())).T
_, Yhat = forward(Xgrid)
ax.plot_trisurf(Xgrid[:,0], Xgrid[:,1], Yhat, linewidth=0.2, antialiased=True)
plt.show()



'''
So the one idea is that we can plot the magnitude of the residuals at each point on the X1,X2 grid
The lite colors mean that it's bad and the lighter color means that it's good
'''
# plot magnitude of residuals
Ygrid = Xgrid[:,0]*Xgrid[:,1]
R = np.abs(Ygrid - Yhat)

plt.scatter(Xgrid[:,0], Xgrid[:,1], c=R)
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(Xgrid[:,0], Xgrid[:,1], R, linewidth=0.2, antialiased=True)
plt.show()
