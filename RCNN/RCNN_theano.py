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

LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "0.01"))
NEPOCH = int(os.environ.get("NEPOCH", "221"))
MODEL_OUTPUT_FILE = os.environ.get("MODEL_OUTPUT_FILE")
PRINT_EVERY = int(os.environ.get("PRINT_EVERY", "5")) 

class RCNN_theano:
    
    def __init__(self):
        self.__theano_build__()
        
    def __theano_build__(self):
        
        
        def forward_recurrent_step(x_tp, x_t, U, W, V, B, BO):
            atp = T.dot(U, x_tp)
            s_tp = T.tanh(atp + B)
            
            at = T.dot(U, x_t)
            b = T.dot(W, s_tp)
            s_t = T.tanh(at + B + b)
            
            o_t = T.clip(T.nnet.sigmoid(T.dot(V, s_t) + BO),0.0000001,0.9999999)
            return [o_t, s_t]