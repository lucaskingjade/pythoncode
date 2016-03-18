import sys
import os
import time
from RNN_theano import RNN_theano
import numpy as np
import theano as theano
import theano.tensor as T
from RNN_theano import train_with_sgd

li = T.matrix('list')
m = T.scalar("m")
n = T.scalar("n")
outf = theano.function([m,n], 2*m + 3*n)
cost = 2*m + 3*n
dm = T.grad(cost, m)
dn = T.grad(cost, n)
rollf = theano.function([li], T.roll(li, -1, 1))
entry = [[0.5,1.1,3],  [0.8,0.1,3] , [0.3,1.5,3] ]
res = rollf(entry)

oll = theano.function([m,n], [dm])
oll2 = theano.function([m,n], [dn])

value = oll(4,5)
value2 = oll2(4,5)

inputd = 2
outputd = 1
hiddend = 20

U1 = np.random.uniform(-np.sqrt(1./inputd), np.sqrt(1./inputd), (hiddend, inputd))
V1 = np.random.uniform(-np.sqrt(1./outputd), np.sqrt(1./outputd), (outputd, hiddend))

U = theano.shared(name='U', value=U1.astype(theano.config.floatX))
V = theano.shared(name='V', value=V1.astype(theano.config.floatX))
   
x = T.matrix('x')
y = T.vector('y')
        
def forward_prop_step(x_t, U, V):
    s_t = T.tanh(T.dot(U, x_t))
    o_t = T.nnet.sigmoid(T.dot(V, s_t))
    #s_t = np.tanh(U.dot(x_t))
    #o_t = softmax(V.dot(s_t))
    return [o_t[0], s_t]

a = np.arange(0,10,0.1)
b = np.arange(5,20,0.1)
xD = np.array([  [   [a[i], b[i]] for i in range(100)  ]   for j in range(2) ])
yD = np.array([  [ [a[i]//6 ] for i in range(100)  ] for j in range(2) ])

[ot, st] = forward_prop_step(x, U, V)

afcuntion = theano.function([x], [ot, st])
nn = xD[0]
#mm = nn.tranpose()
entry = [[0.5],[1.1]]
value = afcuntion(entry)
print("V1")
print(value)

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def compute(x_t, U, V):
    s_t = [[np.tanh(U.dot(i))] for i in x_t]
    o_t = softmax(V.dot(s_t))
    return [o_t[0], s_t]

entry2 = [[[0.5],[1.1]],  [[0.8],[0.1]] , [[0.3],[1.5]] ]
value2 = compute(entry2, U1, V1)
print("V2")
print(value2)

