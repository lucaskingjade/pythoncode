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

class RNNLayer:
    #inputshape (t_step, dimfeature)    outputshape(t_step, outfeature)
    def __init__(self, input, inputshape, hiddendim, outputshape):

        def forward_recurrent_step(x_tp, x_t, U, W, V, B, BO):
            atp = T.dot(U, x_tp)
            s_tp = T.tanh(atp + B)

            at = T.dot(U, x_t)
            b = T.dot(W, s_tp)
            s_t = T.tanh(at + B + b)

            o_t = T.clip(T.nnet.sigmoid(T.dot(V, s_t) + BO),0.0000001,0.9999999)
            return [o_t, s_t]

        assert(inputshape[0] == outputshape[0])
        self.input = input
        self.input_dim = inputshape[1] # + 1
        self.output_dim = outputshape[1]
        self.hidden_dim = hiddendim

        self.U = theano.shared(
            np.random.uniform(-np.sqrt(1./self.input_dim), np.sqrt(1./self.input_dim), (self.hidden_dim, self.input_dim)),
            dtype=theano.config.floatX, name = 'U')
        self.V = theano.shared(
            np.random.uniform(-np.sqrt(1./self.output_dim), np.sqrt(1./self.output_dim), (self.output_dim, self.hidden_dim)),
            dtype=theano.config.floatX, name = 'V')
        self.W = theano.shared(
            np.random.uniform(-np.sqrt(1./self.hidden_dim), np.sqrt(1./self.hidden_dim), (self.hidden_dim, self.hidden_dim)),
            dtype=theano.config.floatX, name = 'W')
        self.B = theano.shared(
            np.random.uniform(-np.sqrt(1./self.hidden_dim), np.sqrt(1./self.hidden_dim), (self.hidden_dim)),
            dtype=theano.config.floatX, name = 'B')
        self.BO = theano.shared(
            np.random.uniform(-np.sqrt(1./self.hidden_dim), np.sqrt(1./self.hidden_dim), (self.output_dim)),
            dtype=theano.config.floatX, name = 'BO')

        [o, st], updates = theano.scan(forward_recurrent_step,
                                      sequence=input,
                                      outputs_info=[None, dict(initial=np.zeros(self.hidden_dim)), None, None],
                                      non_sequences=[self.U, self.W, self.V, self.B, self.BO],
                                      n_steps=inputshape[0])
        self.output = o

    def binary_crossentropy(self, y):
        return T.sum(T.nnet.binary_crossentropy(self.output, y))