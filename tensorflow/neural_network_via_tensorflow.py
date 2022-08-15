# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import matplotlib.pyplot as plt

#let's create 500 samples per class
#So what we are gonna do is generate some gaussian clouds
Nclass = 500
#So we are gonna have 3 gaussian clouds
#np.random.randn(Nclass, 2) + np.array([0,-2]) So the 1st gaussian cloud is gonna centered at 0,-2
X1 = np.random.randn(Nclass, 2) + np.array([0,-2])
#np.random.randn(Nclass, 2) + np.array([2,2]) So the 2nd gaussian cloud is gonna centered at 2,2
X2 = np.random.randn(Nclass, 2) + np.array([2,2])
#np.random.randn(Nclass, 2) + np.array([-2,2]) So the 3rd gaussian cloud is gonna centered at -2,2
X3 = np.random.randn(Nclass, 2) + np.array([-2,2])
X = np.vstack([X1,X2,X3]).astype(np.float32)

#create our labels
Y = np.array([0]*Nclass + [1]*Nclass + [2]*Nclass)

#visualizing the created data points on matplotlib
plt.scatter(X[:,0], X[:,1], c=Y, s=100, alpha=0.5)
plt.show()

#Number of inputs
D=2
#Number of hidden layers
M=3
#number of classes
K=3

#Indicator variables for the targets
N = len(Y)
# turn Y into an indicator matrix for training
T = np.zeros((N,K))
for i in range(N):
    T[i,Y[i]] = 1

#Tensorflow does not use regular numpy arrays instead it uses tensorflow variables
#In this function we are going to initialize some weights also known as parameters
#We have one argument to the function which is the shape of the weights
#Inside this function we want to wrap the return value in a if.Variable() object these are used to encapsulate model parameters,
#parameters that you would want to be trained during back propogation
#inside this td.Variable(tf.random_normal()). tf.random_normal()-> works very similarly to numpy random normal function except it returns a tensorflow object instead of a numpy array
#stddev = standard deviation
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape,stddev=0.01))

'''
This function will describe how the neural network is gonna do forward propogation
This function holds the code which defines how a neural network computes an output
'''
def forward(X,W1,b1,W2,b2):
    #First we will calculate the hidden layer This will use tensorflow version og the sigmoid function
    #inside that we will use tensorflows version og matrix multiplication
    Z = tf.nn.sigmoid(tf.matmul(X,W1) + b1)
    #Note that we do not apply the final softmax at this time the reason for this is numeric stability
    #tensorflow combines the softmax with the loss function
    output = tf.matmul(Z,W2) + b2
    return output

'''
create the tensorflow placeholders These are what represents our data. Notice one important thing which is that this doesn't actually use any
of our data. Our data is in the variable X and Y which we had defined earlier.
Remember all we are doing this time is creating a computational graph. We are only telling tensorflow how to compute things without actually doing any computation.
So here we are gonna pass some basic information into the place holder
-> First we say that the type of data we are going to pass in is float 32
-> Second we say that the shape of the data that we are going to pass in is [None,D] OR None by D. The reason the first value is None because
   we want to be able to pass in data of different sizes
'''
tfX = tf.placeholder(tf.float32, [None,D]) #input
tfY = tf.placeholder(tf.float32, [None,K]) #output OR Targets

'''
So next we are going to create all the parameters for our model using our init_weight function
'''
W1 = init_weights([D,M])
b1 = init_weights([M])
W2 = init_weights([M,K])
b2 = init_weights([K])

#we can use our forward function to get the output of our neural network
'''Note : SO far we have done zero computation. We are only saying how the nesorflow should compute the logits. We have not yet passed in any data'''
logits = forward(tfX, W1,b1,W2,b2)

'''
Next we are going to define our cost or a loss function. this is going to be wraped in a function called tf.reduce_mean() but Why?
Because the loss function we are going to call computes the loss of per sample. So that's why we are gonna take them all togeather
and calculate the mean
'''
#cost function
cost = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits_v2(
    labels = tfY,
    logits = logits,
    )
)

'''
In tensorflow you do not have to do any backpropogation yourself. Tensorflow does all of that automatically
We can setup a training process by setting up an optimizer
Again remember this function hasn't done any computation so far we have only stated how to do the computation later when we actually do them.
'''
train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost)

'''
Define the predict operation
Here also instead of using numpy argmax we are using tensorflow argmax
Again here also no computation is being done. we are lony saying how it should be done
'''
predic_op = tf.argmax(logits, 1)

'''
Finally the last step is to actually do all the computations. Out computational graph is now set so we can actually use it.
'''
sess = tf.compat.v1.Session()
init = tf.global_variables_initializer()
'''
Now we are going to give them actual values when we call session.run(init)
'''
sess.run(init)

'''
create a main training loop
'''
for i in range(1000):
    '''
    Inside the training loop we are going to do one back propogation step
    at this point we pass in our data
    '''
    sess.run(train_op, feed_dict = {tfX:X, tfY:T})

    '''
    The next step is to get our predictions
    '''
    pred = sess.run(predic_op, feed_dict={tfX:X})

    '''
    check if the iteration number is divisible by 100
    '''
    if i %100 == 0:
        print(f"Accuracy : {np.mean(Y==pred)}")
