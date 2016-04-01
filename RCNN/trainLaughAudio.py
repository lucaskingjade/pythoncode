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
from util import *


timestart = datetime.now()
#npx, npy= buildData('laugh')
npx, npy= buildData(datapath = 'final/', type='geste')

orgnizeddatainput, orgnizeddataoutput = prepareData(npx, npy, 20, 4)

#originalLabel = np.argmax(orgnizeddataoutput, axis = 1)
print("data size: %s " %  (len(orgnizeddatainput)))

timeend = datetime.now()
print("data loading: %s second" %  (timeend - timestart).total_seconds())

model = RNN(80, 10, 4)

#acc = calculateAccuracy(model, orgnizeddatainput[0], orgnizeddataoutput[0])
#p = model.predict(orgnizeddatainput[0])

def train_with_sgd_cross(model, X_train, Y_train, learning_rate=LEARNING_RATE, nepoch=NEPOCH, writelog=True, saveparameter = False):
    
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
            num_examples_seen += 1
            if(num_examples_seen % 20 == 0):
                print("output error")
                print(o_error)

        if(epoch % 1 == 0):
            loss = model.calculate_loss(X_train, Y_train)
            losses.append((num_examples_seen, loss))
            if(writelog):
                log_OUTPUT_FILE = "RNN_log.log"
                str = '%s \n'% (loss)
                writeFile(log_OUTPUT_FILE,str)
                
            time = (datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            print ("%s: Loss after num_examples_seen=%d epoch=%d: %f" % (time, num_examples_seen, epoch, loss))
            # Adjust the learning rate if loss increases
            if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
                learning_rate = learning_rate * 0.5
                print ("Setting learning rate to %f" % learning_rate)
            #if(abs(losses[-2][1] - losses[-1][1]) < losses[-1][1] * 0.001):
             #   print("difference error is small")
            if(saveparameter):
                model.layer.save_model_parameters_theano(MODEL_OUTPUT_FILE)
            print("learning rate: %s" % learning_rate)
            
        timeloop = datetime.now()
        
        print("%s epoch: %s second" %  (epoch, (timeloop - timestart).total_seconds()))
        if(writelog):
            acc = calculateAccuracy(model, X_train[10], Y_train[10])
            writeFile('speedlog.log',"[%s epoch] [ %s seconds] [learning rate: %s] [accuray: %s] \n" %  (epoch, (timeloop - timestart).total_seconds(), learning_rate, acc))
            
    timeend = datetime.now()
    print("total train: %s second" %  (timeend - timebegin).total_seconds())
    if(writelog):
        writeFile('speedlog.log', "total train: %s second" %  (timeend - timebegin).total_seconds())
        
    print('training finish')
    
train_with_sgd_cross(model, orgnizeddatainput, orgnizeddataoutput)