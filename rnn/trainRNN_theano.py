import sys
import os
import time
from RNN_theano import RNN_theano
import numpy as np
import theano as theano
import theano.tensor as T
from RNN_theano import train_with_sgd
from RNN_theano import *

# enable on-the-fly graph computations
#theano.config.compute_test_value = 'warn'
THEANO_FLAGS='optimizer=None'

inputd = 2
outputd = 1
hiddend = 5

#def computecrossentry(o,y):
#    return [-(y[i] * math.log(o[i]) + (1 - y[i]) * math.log(1 - o[i])) for i in range(len(y))] 

a = np.arange(0,10,0.1)
b = np.arange(5,20,0.1)
bias = [np.random.sample()/ 100.0 for i in range(100)] 
bias2 = [np.random.sample()/ 100.0 for i in range(100)] 
bias3 = [np.random.sample()/ 100.0 for i in range(100)] 
x = np.array([  [   [a[i] * 0.1 + np.random.sample()/ 100.0, (b[i]) * 0.1 - np.random.sample()/ 100.0] for i in range(100)  ]   for j in range(2) ])
y = np.array([  [ [x[0][i][0] //0.5 ] for i in range(100)  ] for j in range(2) ])

#v = computecrossentry(y[0] , y[0])

#x = np.array([ [ [  np.random.sample(), np.random.sample()] for i in range(100)  ]   for j in range(2) ])
#y = np.array([  [ [x[j][i][0] * x[j][i][1] //0.5 ] for i in range(100)  ] for j in range(2) ])
#forward_propagationX = theano.function([x, U, V], forward_prop_step)

#[o_t, s_t] = forward_propagationX(x, U, V)
# Build model
model = RNN_theano(inputd, outputd, hiddend, bptt_truncate=1)
bptt_gradients = model.bptt(x[0],y[0])
U, V, W = model.U.get_value(), model.V.get_value(), model.W.get_value()

print (U)
print (W)
print (V)

#[y_out, s, a, b, c] = model.forward_propagation(x[0])
# print("y")
# print(y_out[2])
# print("s")
# print(s[2])
# print("a")
# print(a[2])
# print("b")
# print(b[2])
# print("c")
# print(c[2])

#print("bptt_gradients")
#print(bptt_gradients)
#bptt_gradients2 =  model.bpttCalculate(x[0],y[0])
#print(bptt_gradients2)
            
            


#model.sgd_step(x[0], y[0], 0.005)
print ("model train one step.")

train_with_sgd(model, x, y)
print ("model train sgd.")

[r,s,sl,a] = model.forward_propagation(x[0])
print ("model forward one step.")
print (r)

re = model.predict(x[0])
precision, recall, f = evaluationF(re, y[0])
v = accuracy(re, y[0])
print ("accuracy")
print (v)
print ("precision %s  recall %s   f %s", (precision, recall, f))