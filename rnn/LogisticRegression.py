#mostly correct logistic regression
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
NEPOCH = int(os.environ.get("NEPOCH", "500"))
MODEL_OUTPUT_FILE = os.environ.get("MODEL_OUTPUT_FILE")

if not MODEL_OUTPUT_FILE:
    ts = datetime.now().strftime("%Y-%m-%d-%H-%M")
    MODEL_OUTPUT_FILE = "RNN-%s-%s-%s-%s.dat" % (ts, 2, 1, 0)

#INPUT_DATA_FILE = os.environ.get("INPUT_DATA_FILE", "./data/reddit-comments-2015.csv")
PRINT_EVERY = int(os.environ.get("PRINT_EVERY", "5"))  ## EPOQUE

class Regression_theano:
     
    def __init__(self):
        # Assign instance variables
        # input dim = d + 1
        self.input_dim = 2
        self.output_dim = 1

        # Randomly initialize the network parameters
        #=======================================================================
        U = np.random.uniform(-np.sqrt(1./self.input_dim), np.sqrt(1./self.input_dim), (self.output_dim, self.input_dim))
        B = np.random.uniform(-np.sqrt(1./self.output_dim), np.sqrt(1./self.output_dim), (self.output_dim))
     
        self.U_NP = U;
        self.B_NP = B;

        # Theano: Created shared variables
        self.U = theano.shared(name='U', value=U.astype(theano.config.floatX))
        self.B = theano.shared(name='B', value=B)
        
         # We store the Theano graph here
        self.theano = {}
        self.__theano_build__()
        

    def __theano_build__(self):
        U, B = self.U, self.B

        x = T.matrix('x')
        y = T.matrix('y')
        
        def forward_prop_step(x_t, U, B):
            a = T.dot(U, x_t)
            s_t = T.tanh(a + B)
            o_t = T.nnet.sigmoid(s_t * 2 + 0.1)
            return [o_t, s_t, a] #[o_t[0], s_t]
        
        [o, s, a], updates = theano.scan(
            forward_prop_step,
            sequences=x,
            outputs_info=[None, None, None],
            non_sequences=[U, B],
            strict=True)
        
        o_error = T.sum(T.nnet.binary_crossentropy(o, y))#T.nnet.binary_crossentropy(o, y).mean()
        #o_error = T.sum(T.abs_(o - y))# T.abs_(o - y).mean()#
        
        # Gradients
        dU = T.grad(o_error, U)
        dB = T.grad(o_error, B)

        
        # Assign functions
        self.forward_propagation = theano.function([x], [o, s, a])
        self.ce_error = theano.function([x, y], o_error)
        self.bptt = theano.function([x, y], [dU, dB])
        
        # SGD
        learning_rate = T.scalar('learning_rate')
        self.sgd_step = theano.function([x,y,learning_rate], [o_error], 
                      updates=[(self.U, self.U - learning_rate * dU),
                              (self.B, self.B - learning_rate * dB)
                              ])
        
        print ("model build finished.")
       
    
    def calculate_total_loss(self, X, Y):
        #--------------------------------------------------------------- sum = 0
        #------------------------------------------------- for x, y in zip(X,Y):
            #-------------------------------------------- v = self.ce_error(x,y)
            #--------------------------------------------------------- print (v)
            #---------------------------------------------------------- sum += v
        #------------------------------------------------------------ return sum
        return np.sum([self.ce_error(x,y) for x,y in zip(X,Y)])
    
    def calculate_loss(self, X, Y):
        # Divide calculate_loss by the number of words
        num_words = np.sum([len(y) for y in Y])
        return self.calculate_total_loss(X,Y)/float(num_words)  
     
    def bpttCalculate(self, x, y):
        T = len(y)
        # Perform forward propagation
        [o, s, a] = self.forward_propagation(x)
        # We accumulate the gradients in these variables
        dLdU = np.zeros(self.U_NP.shape)
        delta_o = o
        #delta_o[np.arange(len(y)), int(y)] -= 1.
        for i in range(T):
            delta_o[i] = delta_o[i] - y[i]
        # For each output backwards...
        for t in np.arange(T)[::-1]:
            dLdU += np.outer(delta_o[t] * s[t].T, x[t])
        return [dLdU]
    
    def predict(self, x):
        y = self.forward_propagation(x)[0]
        return y // 0.5
    
def gradient_check_theano(model, x, y, h=0.00001, error_threshold=0.01):
    # Overwrite the bptt attribute. We need to backpropagate all the way to get the correct gradient
    
    [y_out,s, a] = model.forward_propagation(x)
    # Calculate the gradients using backprop
    bptt_gradients = model.bptt(x, y)
    
    bptt_gradients2 = model.bpttCalculate(x, y)
    # List of all parameters we want to chec.
    model_parameters = ['U', 'B']
    # Gradient check for each parameter
    for pidx, pname in enumerate(model_parameters):
        #if (pname != 'V'):
        #    continue
        # Get the actual parameter value from the mode, e.g. model.W
        parameter_T = operator.attrgetter(pname)(model)
        parameter = parameter_T.get_value()
        print (("Performing gradient check for parameter %s with size %d." % (pname, np.prod(parameter.shape))))
        # Iterate over each element of the parameter matrix, e.g. (0,0), (0,1), ...
        it = np.nditer(parameter, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            ix = it.multi_index
            # Save the original value so we can reset it later
            original_value = parameter[ix]
            # Estimate the gradient using (f(x+h) - f(x-h))/(2*h)
            parameter[ix] = original_value + h
            parameter_T.set_value(parameter)
            gradplus = model.calculate_total_loss([x],[y])
            parameter[ix] = original_value - h
            parameter_T.set_value(parameter)
            gradminus = model.calculate_total_loss([x],[y])
            estimated_gradient = (gradplus - gradminus)/(2*h)
            parameter[ix] = original_value
            parameter_T.set_value(parameter)
            # The gradient for this parameter calculated using backpropagation
            backprop_gradient = bptt_gradients[pidx][ix]
            # calculate The relative error: (|x - y|/(|x| + |y|))
            relative_error = np.abs(backprop_gradient - estimated_gradient)/(np.abs(backprop_gradient) + np.abs(estimated_gradient))
            # If the error is to large fail the gradient check
            if relative_error > error_threshold:
                print ("Gradient Check ERROR: parameter=%s ix=%s" % (pname, ix))
                print ("+h Loss: %f" % gradplus)
                print ("-h Loss: %f" % gradminus)
                print ("Estimated_gradient: %f" % estimated_gradient)
                print ("Backpropagation gradient: %f" % backprop_gradient)
                print ("Relative Error: %f" % relative_error)
                return 
            it.iternext()
        print ("Gradient check for parameter %s passed." % (pname))
     
    
    
     
    
# Outer SGD Loop
# - model: The RNN model instance
# - X_train: The training data set
# - y_train: The training data labels
# - learning_rate: Initial learning rate for SGD
# - nepoch: Number of times to iterate through the complete dataset
# - evaluate_loss_after: Evaluate the loss after this many epochs
def train_with_sgd(model, X_train, y_train, learning_rate=LEARNING_RATE, nepoch=NEPOCH, evaluate_loss_after=PRINT_EVERY):
    # We keep track of the losses so we can plot them later
    losses = []
    num_examples_seen = 0
    for epoch in range(nepoch):
        # Optionally evaluate the loss
        if (epoch % evaluate_loss_after == 0):
            loss = model.calculate_loss(X_train, y_train)
            losses.append((num_examples_seen, loss))
            time = (datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            print ("%s: Loss after num_examples_seen=%d epoch=%d: %f" % (time, num_examples_seen, epoch, loss))
            # Adjust the learning rate if loss increases
            if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
                learning_rate = learning_rate * 0.5 
                print ("Setting learning rate to %f" % learning_rate)
            sys.stdout.flush()
        # For each training example...
        for i in range(len(y_train)):
            # One SGD step
            o_error = model.sgd_step(X_train[i], y_train[i], learning_rate)
            print("output error %s  learning rate %s", (o_error,learning_rate))
            if(o_error[0] > 300 or o_error[0]<-300 or math.isnan(o_error[0])):
                print(o_error[0])
            #gradient_check_theano(model, X_train[i], y_train[i])
            num_examples_seen += 1
        #save Parameter
        save_model_parameters_theano(MODEL_OUTPUT_FILE, model);
    
    
def save_model_parameters_theano(outfile, model):
    U, B = model.U.get_value(), model.B.get_value()
    np.savez(outfile, U=U, B=B)
    #print ("Saved model parameters to %s." % outfile)
   
def load_model_parameters_theano(path, model):
    npzfile = np.load(path)
    U, B = npzfile["U"], npzfile["B"]
    model.input_dim = U.shape[0]
    model.output_dim = U.shape[1]
    model.U.set_value(U)
    model.B.set_value(B)
    #print ("Loaded model parameters from %s. hidden_dim=%d word_dim=%d" % (path, U.shape[0], B.shape[1]))
            

def compute_prediction(model, test_set):
    return model.forward_propagation(test_set)

def accuracy(predicted, actual):
    total = 0.0
    correct = 0.0
    for p, a in zip(predicted, actual):
        total += 1
        if abs(p - a) < 0.5:
            correct += 1
    return correct / total


#x = np.array([ [ [  np.random.sample(), np.random.sample()] for i in range(100)  ]   for j in range(2) ])
#y = np.array([  [ [x[j][i][0] * x[j][i][1] //0.5 ] for i in range(100)  ] for j in range(2) ])
a = np.arange(0,10,0.1)
b = np.arange(5,20,0.1)
x = np.array([  [   [a[i] * 0.1 + np.random.sample()/ 100.0, (b[i]) * 0.1 - np.random.sample()/ 100.0] for i in range(100)  ]   for j in range(2) ])
y = np.array([  [ [x[0][i][0] //0.5 ] for i in range(100)  ] for j in range(2) ])

model = Regression_theano()
train_with_sgd(model, x, y)
p0 = model.forward_propagation(x[0])[0]
p = model.predict(x[0])
print(p)
print(y[0])
print("accuracy: %s", accuracy(p, y[0]))
print("end")

            