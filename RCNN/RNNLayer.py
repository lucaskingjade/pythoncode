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
    def __init__(self, input, inputdim, hiddendim, outputdim):

        def forward_recurrent_step(x_t, s_t_p, U, W, V, B, BO):
            a = T.dot(U, x_t)
            b = T.dot(W, s_t_p)
            s_t = T.tanh(a + B + b)
            o_t = T.clip(T.nnet.sigmoid(T.dot(V, s_t) + BO), 0.0000001,0.9999999)
            return [o_t, s_t]

        self.input = input
        self.input_dim = inputdim # + 1
        self.output_dim = outputdim
        self.hidden_dim = hiddendim

        self.U = theano.shared(
            np.random.uniform(-np.sqrt(1./self.input_dim), np.sqrt(1./self.input_dim), (self.hidden_dim, self.input_dim)),
             name = 'U')
        self.V = theano.shared(
            np.random.uniform(-np.sqrt(1./self.output_dim), np.sqrt(1./self.output_dim), (self.output_dim, self.hidden_dim)),
             name = 'V')
        self.W = theano.shared(
            np.random.uniform(-np.sqrt(1./self.hidden_dim), np.sqrt(1./self.hidden_dim), (self.hidden_dim, self.hidden_dim)),
             name = 'W')
        self.B = theano.shared(
            np.random.uniform(-np.sqrt(1./self.hidden_dim), np.sqrt(1./self.hidden_dim), (self.hidden_dim)),
             name = 'B')
        self.BO = theano.shared(
            np.random.uniform(-np.sqrt(1./self.hidden_dim), np.sqrt(1./self.hidden_dim), (self.output_dim)),
             name = 'BO')

        # store parameters of this layer
        self.params = [self.U, self.V, self.W, self.B,self.BO]
        [self.output, self.st], updates = theano.scan(forward_recurrent_step,
                                      sequences=input,
                                      outputs_info=[None, dict(initial=np.zeros(self.hidden_dim))],
                                      non_sequences=[self.U, self.W, self.V, self.B, self.BO],
                                      strict=True)

    def binary_crossentropy(self, output, y):
        return T.sum(T.nnet.binary_crossentropy(output, y))
    
    def save_model_parameters_theano(self, outfile):
        U, V, W, B, BO = self.U.get_value(), self.V.get_value(), self.W.get_value(), self.B.get_value(), self.BO.get_value()
        np.savez(outfile, U=U, V=V, W=W, B=B, BO=BO)
        print ("Saved model parameters to %s." % outfile)
   
    def load_model_parameters_theano(self, path):
        npzfile = np.load(path)
        U, V, W, B, BO = npzfile["U"], npzfile["V"], npzfile["W"], npzfile["B"], npzfile["BO"]
        self.hidden_dim = U.shape[0]
        self.U.set_value(U)
        self.V.set_value(V)
        self.W.set_value(W)
        self.B.set_value(B)
        self.BO.set_value(BO)
        print ("load model parameters to %s." % path)


LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "0.001"))
NEPOCH = int(os.environ.get("NEPOCH", "51"))
MODEL_OUTPUT_FILE = os.environ.get("MODEL_OUTPUT_FILE")
PRINT_EVERY = int(os.environ.get("PRINT_EVERY", "10"))
VALID_EVERY = int(os.environ.get("VALID_EVERY", "5"))
SAVE_EVERY = int(os.environ.get("SAVE_EVERY", "2"))

MODEL_OUTPUT_FILE = os.environ.get("MODEL_OUTPUT_FILE")
if not MODEL_OUTPUT_FILE:
    ts = datetime.now().strftime("%Y-%m-%d-%H-%M")
    MODEL_OUTPUT_FILE = "RNN-%s.dat" % (ts)

class RNN:
    def __init__(self, inputdim, hiddendim, outputdim):
        timestart = datetime.now()
        x = T.matrix('x')   # the data is presented as rasterized images
        y = T.matrix('y')  # the labels are presented as 1D vector of
                        # [int] labels

        self.learning_rate = T.scalar('learning_rate')

        self.layer = RNNLayer(x, inputdim, hiddendim, outputdim)

        o_error = self.layer.binary_crossentropy(self.layer.output, y)

        grads = T.grad(o_error, self.layer.params)

        # train_model is a function that updates the model parameters by
        # SGD Since this model has many parameters, it would be tedious to
        # manually create an update rule for each model parameter. We thus
        # create the updates list by automatically looping over all
        # (params[i], grads[i]) pairs.

        updates = [
            (param_i, param_i - self.learning_rate * grad_i)
            for param_i, grad_i in zip(self.layer.params, grads)
        ]

        self.forward_propagation = theano.function([x], [self.layer.output])
        self.bptt = theano.function([x, y], grads)
        self.ce_error = theano.function([x, y], o_error)

        self.sgd_step = theano.function([x,y, self.learning_rate], [o_error],
                      updates=updates)
        
        timeend = datetime.now()
        
        print("model building: %s second" % (timeend - timestart).total_seconds())

    def calculate_total_loss(self, X, Y):
        return np.sum([self.ce_error(x,y) for x,y in zip(X,Y)])

    def calculate_loss(self, X, Y):
        # Divide calculate_loss by the number of words
        num_words = np.sum([len(y) for y in Y])
        return self.calculate_total_loss(X,Y)/float(num_words)

    def predict(self, x):
        y = self.forward_propagation(x)[0]
        return y // 0.5
    
    def loadFile(self, path):
        self.layer.load_model_parameters_theano(path)

def train_with_sgd(model, X_train, Y_train, learning_rate=LEARNING_RATE, nepoch=NEPOCH, evaluate_loss_after=PRINT_EVERY):
    
    flog = open('logEvaluation.txt', 'a')

    # We keep track of the losses so we can plot them later
    timebegin = datetime.now()
    losses = []
    num_examples_seen = 0
    trainDatalen = int(len(Y_train) * 0.6)
    for epoch in range(nepoch):
        timestart = datetime.now()
        # Optionally evaluate the loss
        if (epoch % evaluate_loss_after == 0):
            loss = model.calculate_loss(X_train, Y_train)
            losses.append((num_examples_seen, loss))
            time = (datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            print ("%s: Loss after num_examples_seen=%d epoch=%d: %f" % (time, num_examples_seen, epoch, loss))
            # Adjust the learning rate if loss increases
            if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
                learning_rate = learning_rate * 0.5
                print ("Setting learning rate to %f" % learning_rate)
            sys.stdout.flush()
        # For each training example...
        for i in range(trainDatalen):
            # One SGD step
            #yo = model.forward_propagation(X_train[i])
            #print(yo[1])
            #bptt_gradients = model.bptt(X_train[i], y_train[i])
            o_error = model.sgd_step(X_train[i], Y_train[i], learning_rate)
            str = '%s, '% (o_error[0])
            flog.write(str)
            if(i % 20 == 0):
                print("output error")
                print(o_error)
            num_examples_seen += 1

        if(epoch % VALID_EVERY == 0):
            if(trainDatalen < len(Y_train)):
                validatTest(model, X_train[0], Y_train[0], 0)
                validatTest(model, X_train[trainDatalen + 1], Y_train[trainDatalen + 1], 1)
                
        if(epoch % SAVE_EVERY == 0):
            model.layer.save_model_parameters_theano(MODEL_OUTPUT_FILE)
            
        timeloop = datetime.now()
        
        print("%s epoch: %s second" %  (epoch, (timeloop - timestart).total_seconds()))
            
    timeend = datetime.now()
    print("total train: %s second" %  (timeend - timebegin).total_seconds())
    flog.close()

def prepareData(datainput, dataoutput, frame_perunit, feature_perframe):
    orgnizeddatainput = []
    orgnizeddataoutput = []
    TShape = datainput.shape
    for i in range(TShape[0]):
        for j in range(TShape[1]):
            framesnb = TShape[1]
            numberofUnit = int(framesnb /  frame_perunit)
            less = framesnb % frame_perunit

            for shift in range(frame_perunit):
                sequence = []
                sequenceout = []
                if (less == 0 or shift < less):
                    numberofUnitlocal = numberofUnit
                else:
                    numberofUnitlocal = numberofUnit - 1
                for n in range(numberofUnitlocal - 1):
                    unit = datainput[i][n *  frame_perunit + shift: (n + 1) * frame_perunit + shift]
                    unitflat = unit.flatten()
                    sequence.append(unitflat)
                    sequenceout.append(dataoutput[i][(n + 1) * frame_perunit + shift])
                orgnizeddatainput.append(sequence)
                orgnizeddataoutput.append(sequenceout)
    return orgnizeddatainput, orgnizeddataoutput

def evaluationF(predicted, actual):
    c1c = 0
    c1o = 0
    c1p = 0
    for p, a in zip(predicted, actual):
        if p >= 0.5 and a >= 0.5:
            c1c += 1
            c1o += 1
            c1p += 1
        elif p < 0.5 and a >= 0.5:
            c1o += 1
        elif p >= 0.5:
            c1p += 1
    precision = c1c/c1p
    recall = c1c/c1o
    f = 2 * (precision * recall) / (precision + recall)
    return precision, recall, f

def accuracy(predicted, actual):
    total = 0.0
    correct = 0.0
    for p, a in zip(predicted, actual):
        total += 1
        if abs(p - a) < 0.5:
            correct += 1
    return correct / total

def validatTest(model, x_valid, y_valid, sig):
     #for i in range(len(y_valid)):
     re = model.predict(x_valid)
     precision, recall, f = evaluationF(re, y_valid)
     if(sig == 0):
        print("training test : precision(%s),  recall(%s), f(%s)" % (precision, recall, f))
     else:
        print("validation test : precision(%s),  recall(%s), f(%s)" % (precision, recall, f))

def test():
    
    #np.savez("aa.txt", U=1, V=1, W=1, B=1, BO=1)
    timestart = datetime.now()
    a = np.arange(0,10,0.1)
    b = np.arange(5,20,0.1)
    bias = [np.random.sample()/ 100.0 for i in range(100)]
    bias2 = [np.random.sample()/ 100.0 for i in range(100)]
    bias3 = [np.random.sample()/ 100.0 for i in range(100)]
    x = np.array([  [   [a[i] * 0.1 + np.random.sample()/ 100.0, (b[i]) * 0.1 - np.random.sample()/ 100.0,] for i in range(100)]   for j in range(5) ])
    y = np.array([  [ [x[0][i][0] //0.5 ] for i in range(100)  ] for j in range(5) ])

    timeend = datetime.now()
    print("data loading: %s second" %  (timeend - timestart).total_seconds())
    
    orgnizeddatainput, orgnizeddataoutput = prepareData(x, y, 2, 2)

    model = RNN(4, 10, 1)

    train_with_sgd(model, orgnizeddatainput, orgnizeddataoutput)
    #orgnizeddatainput, orgnizeddataoutput = prepareData(x, y, 2, 2)


test()