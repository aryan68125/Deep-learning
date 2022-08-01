import numpy as np

#let's just pretend that this is the output of a neural network
# Or the activation at the last layer
'''
So we have 5 different numbers that represent the activation at 5 different nodes at the output layer and now we
want to do softmax on these numbers
'''
a=np.random.randn(5)
print(f"the activation at 5 different nodes at the output layer : {a}")

'''
perform softmax operation on the activation
step 1 : exponentiate these activation
step 2 : divide by the sum of expa
'''
expa = np.exp(a)
print(f"activation after exponentiation : {expa}")

#The final result of the softmax
answer = expa/expa.sum()
print(f"step 2 answer after dividing expa/expa.sum(): {answer}")

#You can confirm that the answer is a probabilities by adding them up and the sum should be 1
print(f"You can confirm that the answer is a probabilities by adding them up and the sum should be 1 : {answer.sum()}")

#-------------------------------------------------------------------------------------------------------------------------------------------

'''
Now as you know we are going to be working with multiple samples at the same time so let's try this again with a matrix
'''
#Array a of 100 samples and 5 classes
A = np.random.randn(100,5)
print(f"Array A of 100 samples and 5 classes : {A}")

'''
perform softmax operation on the activation
step 1 : exponentiate these activation (So that they are all positive)
step 2 : divide by the sum
         Now you are gonna have a problem here because you dont want to divide by the whole sum because then the whole thing is gonna sum to 1
         "answer = expA/expA.sum()"
         We want every row to sum to one for each sample. axis=1 means we want to do summision along the row
         "answer = expA/expA.sum(axis=1,keepdims=True)"
'''
expA = np.exp(A)
print(f"activation after exponentiation : {expA}")

#The final result of the softmax
Result = expA/expA.sum(axis=1,keepdims=True)
print(f"step 2 answer after dividing expa/expa.sum(): {Result}")

#You can confirm that the result is a probabilities by adding them up and the sum should be 100 because each row summision should be 1 and there are
#100 samples so the final summision should be -> 100
print(f"You can confirm that the answer is a probabilities by adding them up and the sum should be 1 : {Result.sum()}")

#sum along rows = 1
print(f"sum along rows : {Result.sum(axis=1)}")
