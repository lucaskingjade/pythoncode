import sys
import os
import time
from RNN_theano import RNN_theano
import numpy as np
import theano as theano
import theano.tensor as T
from RNN_theano import train_with_sgd

inputd = 2
outputd = 1
hiddend = 20

U = np.random.uniform(-np.sqrt(1./inputd), np.sqrt(1./inputd), (hiddend, inputd))
V = np.random.uniform(-np.sqrt(1./outputd), np.sqrt(1./outputd), (outputd, hiddend))

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def forward_prop_step(x_t, U, V):
    #s_t = T.tanh(T.dot(U, x_t))
    #o_t = T.nnet.softmax(T.dot(V, s_t))
    s_t = np.tanh(U.dot(x_t))
    o_t = softmax(V.dot(s_t))
    return [o_t[0], s_t]


a = np.arange(0,10,0.1)
b = np.arange(5,20,0.1)
x = np.array([  [   [a[i], b[i]] for i in range(100)  ]   for j in range(2) ])
#x2 = [np.array([a[i] for i in range(5)])]

#y = np.array([  [ b[i]//10 for i in range(100)  ] for j in range(2) ])
y = np.array([  [ [a[i]//6 ] for i in range(100)  ] for j in range(2) ])

[o_t, s_t] = forward_prop_step(x[0].transpose(), U, V)
print(o_t)
print(s_t)