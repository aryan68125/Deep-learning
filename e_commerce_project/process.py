import numpy as np
import pandas as pd

def get_data():
    #read the csv file from the file manager using pandas library
    df = pd.read_csv('ecommerce_data.csv')
    #turn the dataFrame as a numpy Matrix
    data = df.values

    #split out X and Y
    #since Y is the last column we will have to exclude the last column when spliting the dataFrame matrix
    # select every row but exclude the last column data[:,:-1]
    X = data[:,:-1]
    #Y is the last column
    #select every row and only the last column by data[:,-1]
    Y = data[:,-1]

    #Normalize the numerical columns
    X[:,1] = (X[:,1]-X[:,1].mean())/X[:,1].std()
    X[:,2] = (X[:,2]-X[:,2].mean())/X[:,2].std()

    #Now let's work on the categorical column which is time of day
    #Get the shape of the original X
    #N = number of samples and D = number of Features
    N,D = X.shape
    #Make a new X and we know this has to be of shape (N,D+3) because there are 4 different categorical values
    X2 = np.zeros((N,D+3))
    #We know that most of the X is gonna be the same
    X2[:,0:(D-1)] = X[:,0:(D-1)]

    #NOW we are going to perform one hot encoding for the other 4 columns
    #Simple way
    #We are gonna loop through every sample
    for n in range(N):
        #we are gonna get the time of day
        #So remember this is either 0,1,2 or 3
        t = int(X[n,D-1])
        #and now we are gonna set this value to X2
        X2[n,t+D-1] = 1
    #Another way
    #create a new matrix of size N and then 4 for the 4 column
    z= np.zeros((N,4))
    #you could index Z directly
    #so in the first dimension you pass arange(N), in the second dimension you pass in the categories
    #you will have to turn them into int first to be safe and you set those specific indicies to 1
    z[np.arange(N),X[:,D-1].astype(np.int32)] = 1
    #sanity check
    assert(np.abs(X2[:,-4:]-z).sum()<10e-10)

    return X2,Y

#For the logistic class we only want the binary data we don't want the full dataset
def get_binary_data():
    #It's gonna take in the data
    X,Y = get_data()
    #It's gonna filter it by only taking classes zeros and one
    X2 = X[Y<=1]
    Y2 = Y[Y<=1]
    return X2,Y2
