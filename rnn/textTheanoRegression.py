import sys
import os
import time
from RNN_theano import RNN_theano
import numpy as np
import theano as theano
import theano.tensor as T

THEANO_FLAGS='optimizer=None'

W_shape = (2, 100)
b_shape = 2
W = theano.shared(np.random.random(W_shape) - 0.5, name="W")
b = theano.shared(np.random.random(b_shape) - 0.5, name="b")

x = T.dmatrix("x") # N x 784
labels = T.dmatrix("labels") # N x 10

output = T.nnet.softmax(T.dot(W, x) + b)

prediction = T.argmax(output, axis=1)

cost = T.nnet.binary_crossentropy(output, labels).mean()

compute_prediction = theano.function([x], prediction)
compute_cost = theano.function([x, labels], cost)

# Compute the gradient of our error function
grad_W = T.grad(cost, W)
grad_b = T.grad(cost, b)

# Set up the updates we want to do
alpha = 2
updates = [(W, W - alpha * grad_W),
           (b, b - alpha * grad_b)]

# Set up the updates we want to do
alpha = T.dscalar("alpha")
updates = [(W, W - alpha * grad_W),
           (b, b - alpha * grad_b)]

# Make our function. Have it return the cost!
train = theano.function([x, labels, alpha],
                 cost,
                 updates=updates)

alpha = 10.0

costs = []

a = np.arange(0,10,0.1)
b = np.arange(5,20,0.1)
x = np.array( [   [a[i], b[i]] for i in range(100)  ]  )
y = np.array(  [ [a[i]//6 ] for i in range(100)  ] )

while True:
    costs.append(float(train(x, y, alpha)))
    
    if len(costs) % 10 == 0:
        print ('Epoch', len(costs), 'with cost', costs[-1], 'and alpha', alpha)
    if len(costs) > 2 and costs[-2] - costs[-1] < 0.0001:
        if alpha < 0.2:
            break
        else:
            alpha = alpha / 1.5