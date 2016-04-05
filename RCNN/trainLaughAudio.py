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

#===============================================================================
# npzfile = np.load('../result/4nodes/rnn_cross/1/parameter2.data.npz')
# U, V, W, B, BO = npzfile["U"], npzfile["V"], npzfile["W"], npzfile["B"], npzfile["BO"]
# print(U)
# 
# npzfile = np.load('../result/4nodes/rnn_cross/2/parameter2.data.npz')
# U, V, W, B, BO = npzfile["U"], npzfile["V"], npzfile["W"], npzfile["B"], npzfile["BO"]
# print(U)
#===============================================================================

timestart = datetime.now()
npx, npy= buildData(datapath = 'final/', type='laugh')
#npx, npy= buildData(datapath = 'final/', type='geste')

frames = 1
features = 4
orgnizeddatainput, orgnizeddataoutput = prepareData(npx, npy, frames, features)

#originalLabel = np.argmax(orgnizeddataoutput, axis = 1)
print("data size: %d " %  (len(orgnizeddatainput)))

timeend = datetime.now()
print("data loading: %f second" %  (timeend - timestart).total_seconds())

#model = RNN(80, [50,10], 4)
model = RNN(frames * features, 4, 4)
model.reinitialParameters()

#acc = calculateAccuracy(model, orgnizeddatainput[0], orgnizeddataoutput[0])
#p = model.predict(orgnizeddatainput[0])

def train_with_sgd_cross(model, X_train, Y_train, learning_rate=LEARNING_RATE, nepoch=NEPOCH, writelog=True, saveparameter = False):
    
    # We keep track of the losses so we can plot them later
    timebegin = datetime.now()
    losses = []
    num_examples_seen = 0
    trainDatalen = int(len(Y_train) * 0.9)
    testlen = int(len(Y_train))
    
    if(writelog):
        acc = []
        for x, y in zip(X_train, Y_train):
            acc.append(calculateAccuracy(model, x, y))
        acctrain = np.mean(acc[0:trainDatalen])
        acctest = np.mean(acc[trainDatalen:testlen])
        writeFile('speedlog.log',"[%d epoch] [ %f seconds] [learning rate: %f] [accuray: %f %f] \n" %  (-1, 0, learning_rate, acctrain, acctest))
        
    
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
                strv = '%f \n'% (loss)
                writeFile(log_OUTPUT_FILE, strv)
                
            time = (datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            print ("%s: Loss after num_examples_seen=%d epoch=%d: loss=%f" % (time, num_examples_seen, epoch, loss))
            # Adjust the learning rate if loss increases
            if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
                learning_rate = learning_rate * 0.5
                print ("Setting learning rate to %f" % learning_rate)
            #if(abs(losses[-2][1] - losses[-1][1]) < losses[-1][1] * 0.001):
            #   print("difference error is small")
            if(saveparameter):
                model.layer.save_model_parameters_theano(MODEL_OUTPUT_FILE)
            print("learning rate: %f" % learning_rate)
            
        timeloop = datetime.now()
        
        print("%d epoch: %f second" %  (epoch, (timeloop - timestart).total_seconds()))
        if(writelog):
            acc = []
            for x, y in zip(X_train, Y_train):
                acc.append(calculateAccuracy(model, x, y))
            acctrain = np.mean(acc[0:trainDatalen])
            acctest = np.mean(acc[trainDatalen:testlen])
            writeFile('speedlog.log',"[%d epoch] [ %f seconds] [learning rate: %f] [accuray: %f %f] \n" %  (epoch, (timeloop - timestart).total_seconds(), learning_rate, acctrain, acctest))
            
    timeend = datetime.now()
    print("total train: %f second" %  (timeend - timebegin).total_seconds())
    if(writelog):
        writeFile('speedlog.log', "total train: %f second \n" %  (timeend - timebegin).total_seconds())
        
    print('training finish')
    
    
def calculateEverageAccuracy(model, X_train, Y_train, idx):
    trainDatalen = int(len(Y_train) * 0.9)
    testlen = int(len(Y_train))
    acc = []
    for x, y in zip(X_train, Y_train):
        acc.append(calculateAccuracy(model, x, y))
    acctrain = np.mean(acc[0:trainDatalen])
    acctest = np.mean(acc[trainDatalen:testlen])
    writeFile('crossvalidation.log', "%d %f %f\n" %  (idx, acctrain, acctest))
    
def train_with_CrossValidation(model, orgnizeddatainput, orgnizeddataoutput):
    tlen = len(orgnizeddatainput)
    arr = np.random.permutation(tlen)
    #ramdom init
    inp = []
    outp = []
    for i in arr:
        inp.append(orgnizeddatainput[i])
        outp.append(orgnizeddataoutput[i])
        
    inputdata = np.asarray(inp)
    outputdata = np.asarray(outp)
    
    shift = int(tlen / 10)
    for i in range(10):
        model.reinitialParameters()
        inputX = np.roll(inputdata, shift)
        inputY = np.roll(outputdata, shift)
        train_with_sgd_cross(model, inputX, inputY, 0.01, 20)
        calculateEverageAccuracy(model, inputX, inputY, i)
        model.saveParametersInFile('parameter%d.data'%(i))
        
    
#train_with_sgd_cross(model, orgnizeddatainput, orgnizeddataoutput)
train_with_CrossValidation(model, orgnizeddatainput, orgnizeddataoutput)