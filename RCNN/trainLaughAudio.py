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
from ReadData import *
from RNNLayer import *


timestart = datetime.now()
npx, npy= buildData('laugh')

orgnizeddatainput, orgnizeddataoutput = prepareData(npx, npy, 50, 4)

timeend = datetime.now()
print("data loading: %s second" %  (timeend - timestart).total_seconds())

model = RNN(200, 10, 4)

def train_with_sgd_cross(model, X_train, Y_train, learning_rate=LEARNING_RATE, nepoch=NEPOCH):
    ts = datetime.now().strftime("%Y-%m-%d-%H-%M")
    log_OUTPUT_FILE = "RNN_log_%s.log" % (ts)
    flog = open(log_OUTPUT_FILE, 'a')

    # We keep track of the losses so we can plot them later
    timebegin = datetime.now()
    losses = []
    num_examples_seen = 0
    trainDatalen = int(len(Y_train) * 0.6)
    for epoch in range(nepoch):
        timestart = datetime.now()
        # For each training example...
        for i in range(trainDatalen):
            # One SGD step
            o_error = model.sgd_step(X_train[i], Y_train[i], learning_rate)
            #y = model.forward_propagation(X_train[i])
            num_examples_seen += 1
            if(num_examples_seen % 100 == 0):
                print("output error")
                print(o_error)

        #if(epoch % VALID_EVERY == 0):
         #   if(trainDatalen < len(Y_train)):
          #      validatTest(model, X_train[0], Y_train[0], 0)
           #     validatTest(model, X_train[trainDatalen + 1], Y_train[trainDatalen + 1], 1)
                
        #if(epoch % SAVE_EVERY == 0):
        if(epoch % 1 == 0):
            loss = model.calculate_loss(X_train, Y_train)
            losses.append((num_examples_seen, loss))
            str = '%s, '% (loss)
            flog.write(str)
            time = (datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            print ("%s: Loss after num_examples_seen=%d epoch=%d: %f" % (time, num_examples_seen, epoch, loss))
            # Adjust the learning rate if loss increases
            if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
                learning_rate = learning_rate * 0.5
                print ("Setting learning rate to %f" % learning_rate)
            sys.stdout.flush()
            #if(abs(losses[-2][1] - losses[-1][1]) < losses[-1][1] * 0.001):
             #   print("difference error is small")
            
            model.layer.save_model_parameters_theano(MODEL_OUTPUT_FILE)
            print("learning rate: %s" % learning_rate)
            
        timeloop = datetime.now()
        
        print("%s epoch: %s second" %  (epoch, (timeloop - timestart).total_seconds()))
            
    timeend = datetime.now()
    print("total train: %s second" %  (timeend - timebegin).total_seconds())
    flog.close()
    
train_with_sgd_cross(model, orgnizeddatainput, orgnizeddataoutput)