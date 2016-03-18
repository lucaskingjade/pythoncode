import sys
import os
import time
import numpy as np
import theano as theano
import theano.tensor as T
import operator
import sys
import math
from datetime import datetime
import RNNLayer
import ConvPoolLayer

LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "0.01"))
NEPOCH = int(os.environ.get("NEPOCH", "221"))
MODEL_OUTPUT_FILE = os.environ.get("MODEL_OUTPUT_FILE")
PRINT_EVERY = int(os.environ.get("PRINT_EVERY", "5")) 

class RCNN_theano:
    
    def __init__(self):
        window = 24
        x = T.matrix('x')
        y = T.ivector('y')
        layer0_input = x.reshape((1, 1, 1, window))

        layer0 = ConvPoolLayer(input=layer0_input,
                               image_shape=(1, 1, 1, window),
                               filter_shape=(1, 1, 1, 5),
                               poolsize=(1, 2))

        layer1 = RNNLayer(input = layer0.output, inputshape=(2,12), hiddendim = 5, outputshape = (2, 1))

        params = layer0.params + layer1.params
        cost = layer1.binary_crossentropy(y)
        grads = T.grad(cost, params)
        updates = [ (param_i, param_i - LEARNING_RATE * grad_i) for param_i, grad_i in zip(params, grads) ]