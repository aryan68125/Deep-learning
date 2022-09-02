'''
In this file we compare the progression of the cost function vs iteration
for 3 cases :
    1) full gradient descent
    2) mini-batch gradient descent
    3) stochastic gradient descent
'''
from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from datetime import datetime

from util import get_normalized_data, forward, error_rate, cost, gradW, gradb, y2indicator

def main():
     #loading in the data
     Xtrain, Xtest, Ytrain, Ytest = get_normalized_data()
     print("Performing logistic regression...")
     N,D=Xtrain.shape

     #convert the targets into indicator matrices using Y2 indicator
     Ytrain_ind = y2indicator(Ytrain)
     Ytest_ind = y2indicator(Ytest)

     #1. full gradient descent
     #2. stochastic gradient descent
     #3. mini batch gradient descent
     '''
     Here we are making a copy for our initial weights this is so that for all three methods we try they will start with
     the same initial weights. This will be more fair since some starting positions may be better or worst than others
     '''
     W = np.random.randn(D,10)/np.sqrt(D)
     W0 = W.copy() # save for later
     b = np.zeros(10)
     test_losses_full = []
     # Other methods cannot use the learning rate this high
     lr = 0.9
     reg = 0.0
     #t0 is the current time
     t0 = datetime.now()
     #This will keep track of the duration from the start to the end of the previous epoch
     last_dt = 0
     '''
     "intervals" is going save the time intervals between each time we compute the test laws we are going to use these to calculate
     the average time intervals.
     This is so that later on we can use the same average time interval to compute the test the loss for the other methods.
     This is so that comparing the other methods would be more fair.
     '''
     intervals = []
     for i in range(50):
         p_y = forward(Xtrain,W,b)
         '''
         Here is a slight difference , here when I calculate the gradient I divide by N the number of samples.
         This is equivalent to dividing the loss by N and then taking the gradient.
         Essentially this means that we are taking the mean instead of the sum to calculate the total loss overall samples.
         The reason we want to do this is so that the gradients for all three methods are on the same scale.
         And the reason we want to do that is so that the learning rates between all three methods are comparable.
         '''
         gW = gradW(Ytrain_ind,p_y,Xtrain)/N
         gb = gradb(Ytrain_ind,p_y)/N

         W+= lr*(gW-reg*W)
         b+= lr*(gb-reg*b)
         '''
         Here we are calculating the test loss importantly after calucating the test loss we assign a new variable called dt.
         dt is the current time minus the current time t0. This gives us the time delta object on which we call the total
         seconds method. There fore dt is the total number of seconds since we have started training.
         '''
         p_y_test = forward(Xtest, W, b)
         test_loss = cost(p_y_test, Ytest_ind)
         dt = (datetime.now() - t0).total_seconds()

         #save these
         '''
         The next step is to save all this information.
         first we calculate the dt2 where dt2 is the difference between dt and the last dt.
         In other words it's the time interval for this particular epoch.
         In other words it's the time it took for this epoch to complete.
         That's why we appended dt2 into the list of intervals.
         also we need to assign the current dt to the last_dt. So the last_dt will store the current value for the next iteration of this loop.
         Next we append the current loss to our list of losses called test_losses_full. Note that because we want the loss per unit time
         we need to store two items specifically we store dt along with the test_loss itself. therefore we saw the test loss along with the time
         since we started training to achieve this test loss.
         '''
         dt2 = dt - last_dt
         last_dt = dt
         intervals.append(dt2)

         test_losses_full.append([dt, test_loss])
         if (i + 1) % 10 == 0:
             print("Cost at iteration %d: %.6f" % (i + 1, test_loss))
     p_y = forward(Xtest, W, b)
     print("Final error rate:", error_rate(p_y, Ytest))
     print("Elapsted time for full GD:", datetime.now() - t0)

     # save the max time so we don't surpass it in subsequent iterations
     max_dt = dt
     avg_interval_dt = np.mean(intervals)


     # 2. stochastic
     W = W0.copy()
     b = np.zeros(10)
     test_losses_sgd = []
     lr = 0.001
     reg = 0.

     t0 = datetime.now()
     last_dt_calculated_loss = 0
     done = False
     for i in range(50): # takes very long since we're computing cost for 41k samples
         tmpX, tmpY = shuffle(Xtrain, Ytrain_ind)
         for n in range(N):
             x = tmpX[n,:].reshape(1,D)
             y = tmpY[n,:].reshape(1,10)
             p_y = forward(x, W, b)

             gW = gradW(y, p_y, x)
             gb = gradb(y, p_y)

             W += lr*(gW - reg*W)
             b += lr*(gb - reg*b)

             dt = (datetime.now() - t0).total_seconds()
             dt2 = dt - last_dt_calculated_loss

             if dt2 > avg_interval_dt:
                 last_dt_calculated_loss = dt
                 p_y_test = forward(Xtest, W, b)
                 test_loss = cost(p_y_test, Ytest_ind)
                 test_losses_sgd.append([dt, test_loss])

             # time to quit
             if dt > max_dt:
                 done = True
                 break
         if done:
             break

         if (i + 1) % 1 == 0:
             print("Cost at iteration %d: %.6f" % (i + 1, test_loss))
     p_y = forward(Xtest, W, b)
     print("Final error rate:", error_rate(p_y, Ytest))
     print("Elapsted time for SGD:", datetime.now() - t0)


     # 3. mini-batch
     W = W0.copy()
     b = np.zeros(10)
     test_losses_batch = []
     batch_sz = 500
     lr = 0.08
     reg = 0.
     n_batches = int(np.ceil(N / batch_sz))


     t0 = datetime.now()
     last_dt_calculated_loss = 0
     done = False
     for i in range(50):
        tmpX, tmpY = shuffle(Xtrain, Ytrain_ind)
        for j in range(n_batches):
            x = tmpX[j*batch_sz:(j + 1)*batch_sz,:]
            y = tmpY[j*batch_sz:(j + 1)*batch_sz,:]
            p_y = forward(x, W, b)

            current_batch_sz = len(x)
            gW = gradW(y, p_y, x) / current_batch_sz
            gb = gradb(y, p_y) / current_batch_sz

            W += lr*(gW - reg*W)
            b += lr*(gb - reg*b)

            dt = (datetime.now() - t0).total_seconds()
            dt2 = dt - last_dt_calculated_loss

            if dt2 > avg_interval_dt:
                last_dt_calculated_loss = dt
                p_y_test = forward(Xtest, W, b)
                test_loss = cost(p_y_test, Ytest_ind)
                test_losses_batch.append([dt, test_loss])

            # time to quit
            if dt > max_dt:
                done = True
                break
        if done:
            break

        if (i + 1) % 10 == 0:
            print("Cost at iteration %d: %.6f" % (i + 1, test_loss))
     p_y = forward(Xtest, W, b)
     print("Final error rate:", error_rate(p_y, Ytest))
     print("Elapsted time for mini-batch GD:", datetime.now() - t0)


     # convert to numpy arrays
     test_losses_full = np.array(test_losses_full)
     test_losses_sgd = np.array(test_losses_sgd)
     test_losses_batch = np.array(test_losses_batch)


     plt.plot(test_losses_full[:,0], test_losses_full[:,1], label="full")
     plt.plot(test_losses_sgd[:,0], test_losses_sgd[:,1], label="sgd")
     plt.plot(test_losses_batch[:,0], test_losses_batch[:,1], label="mini-batch")
     plt.legend()
     plt.show()



if __name__ == '__main__':
    main()
