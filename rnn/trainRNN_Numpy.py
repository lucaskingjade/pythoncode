import sys
import os
import time
import numpy as np
from RNNNumpy import RNNNumpy
from RNNNumpy import *

inputd = 2
outputd = 1
hiddend = 5


a = np.arange(0,10,0.1)
b = np.arange(5,20,0.1)
x = np.array([  [   [a[i] * 0.1, (b[i]) * 0.1] for i in range(100)  ]   for j in range(2) ])
y = np.array([  [ [x[0][i][0] //0.5 ] for i in range(100)  ] for j in range(2) ])


# Build model
model = RNNNumpy(inputd, outputd, hiddend, bptt_truncate=4)
[o, s] = model.forward_propagation(x[0])
#print("o")
#print(o)
# print("s")
# print(s)

#model.numpy_sdg_step(x[0], y[0], 0.01)
train_with_sgd(model, x, y)

print("end")