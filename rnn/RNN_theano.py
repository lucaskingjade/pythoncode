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
from termcolor import colored

LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "0.01"))

INPUT_DIM = int(os.environ.get("INPUT_DIM", "2"))
HIDDEN_DIM = int(os.environ.get("HIDDEN_DIM", "2"))
OUTPUT_DIM = int(os.environ.get("OUTPUT_DIM", "1"))
BPTT_TRUNCATE = int(os.environ.get("BPTT_TRUNCATE", "1"))

NEPOCH = int(os.environ.get("NEPOCH", "221"))
MODEL_OUTPUT_FILE = os.environ.get("MODEL_OUTPUT_FILE")

#INPUT_DATA_FILE = os.environ.get("INPUT_DATA_FILE", "./data/reddit-comments-2015.csv")
PRINT_EVERY = int(os.environ.get("PRINT_EVERY", "5"))  ## EPOQUE

if not MODEL_OUTPUT_FILE:
    ts = datetime.now().strftime("%Y-%m-%d-%H-%M")
    MODEL_OUTPUT_FILE = "RNN-%s-%s-%s-%s.dat" % (ts, INPUT_DIM, OUTPUT_DIM, HIDDEN_DIM)

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

class RNN_theano:
     
    def __init__(self, input_fdim = INPUT_DIM, output_ldim = OUTPUT_DIM, hidden_dim=HIDDEN_DIM, bptt_truncate=BPTT_TRUNCATE):
        # Assign instance variables
        # input dim = d + 1
        self.input_dim = input_fdim # + 1  
        self.output_dim = output_ldim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        
        # Randomly initialize the network parameters
        #=======================================================================
        U = np.random.uniform(-np.sqrt(1./self.input_dim), np.sqrt(1./self.input_dim), (self.hidden_dim, self.input_dim))
        B = np.random.uniform(-np.sqrt(1./self.hidden_dim), np.sqrt(1./self.hidden_dim), (self.hidden_dim))
        BO = np.random.uniform(-np.sqrt(1./self.output_dim), np.sqrt(1./self.output_dim), self.output_dim)
        V = np.random.uniform(-np.sqrt(1./self.output_dim), np.sqrt(1./self.output_dim), (self.output_dim, self.hidden_dim))
        W = np.random.uniform(-np.sqrt(1./self.hidden_dim), np.sqrt(1./self.hidden_dim), (self.hidden_dim, self.hidden_dim))
        #=======================================================================
        
        #=======================================================================
        # U = np.random.uniform(-np.sqrt(self.input_dim), np.sqrt(self.input_dim), (self.hidden_dim, self.input_dim))
        # B = np.random.uniform(-np.sqrt(self.hidden_dim), np.sqrt(self.hidden_dim), (self.hidden_dim))
        # BO = np.random.uniform(-np.sqrt(self.output_dim), np.sqrt(self.output_dim), (self.output_dim))
        # V = np.random.uniform(-np.sqrt(self.output_dim), np.sqrt(self.output_dim), (self.output_dim, self.hidden_dim))
        # W = np.random.uniform(-np.sqrt(self.hidden_dim), np.sqrt(self.hidden_dim), (self.hidden_dim, self.hidden_dim))
        #=======================================================================
        #U, B, BO, V, W = U * 10, B * 10, BO * 10, V * 10, W * 10
        
        self.U_NP = U;
        self.B_NP = B;
        self.BO_NP = BO;
        self.V_NP = V;
        self.W_NP = W;

        # Theano: Created shared variables
        self.U = theano.shared(name='U', value=U.astype(theano.config.floatX))
        self.B = theano.shared(name='B', value=B)
        self.BO = theano.shared(name='BO', value=BO)
        self.V = theano.shared(name='V', value=V.astype(theano.config.floatX))
        self.W = theano.shared(name='W', value=W.astype(theano.config.floatX))  
        
         # We store the Theano graph here
        self.theano = {}
        self.__theano_build__()
        

    def __theano_build__(self):
        U, V, W, B, BO = self.U, self.V, self.W, self.B, self.BO
        
       # st_v = np.zeros(self.bptt_truncate, self.hidden_dim)
        #st_shared = theano.shared(name='st', value=st_v.astype(theano.config.floatX))
        
        x = T.matrix('x')
        y = T.matrix('y')
        
        def forward_prop_step(x_t, s_t_p, U, W, V, B, BO):
            a = T.dot(U, x_t)
            b = T.dot(W, s_t_p)
            s_t = T.tanh(a + B + b)
            o_t = T.clip(T.nnet.sigmoid(T.dot(V, s_t) + BO),0.0000001,0.9999999)
            return [o_t, s_t, s_t_p, a] #[o_t[0], s_t]
        
        [o, s, s_l, a], updates = theano.scan(
            forward_prop_step,
            sequences=x,
            #truncate_gradient=self.bptt_truncate,
            outputs_info=[None, dict(initial=np.zeros(self.hidden_dim)), None, None],
            non_sequences=[U, W, V, B, BO],
            strict=True)
        

        o_error = T.sum(T.nnet.binary_crossentropy(o, y))#T.nnet.binary_crossentropy(o, y).mean()
        #o_error = T.sum(T.abs_(o - y))# T.abs_(o - y).mean()#
        
        # Gradients
        dW = T.grad(o_error, W)
        dU = T.grad(o_error, U)
        dV = T.grad(o_error, V)
        dB = T.grad(o_error, B)
        dBO = T.grad(o_error, BO)

        # Assign functions
        self.forward_propagation = theano.function([x], [o, s, s_l, a])
        self.ce_error = theano.function([x, y], o_error)
        #self.bptt = theano.function([x, y], [dU, dV, dW])
        self.bptt = theano.function([x, y], [dU, dV, dW, dB, dBO])
        
        # SGD
        learning_rate = T.scalar('learning_rate')
        self.sgd_step = theano.function([x,y,learning_rate], [o_error], 
                      updates=[(self.U, self.U - learning_rate * dU),
                               (self.V, self.V - learning_rate * dV),
                               (self.W, self.W - learning_rate * dW),
                              (self.B, self.B - learning_rate * dB),
                              (self.BO, self.BO - learning_rate * dBO)
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
        [o, s, sl, a] = self.forward_propagation(x)
        # We accumulate the gradients in these variables
        dLdU = np.zeros(self.U_NP.shape)
        dLdV = np.zeros(self.V_NP.shape)
        dLdW = np.zeros(self.W_NP.shape)
        delta_o = o
        #delta_o[np.arange(len(y)), int(y)] -= 1.
        for i in range(T):
            delta_o[i] = delta_o[i] - y[i]
        # For each output backwards...
        for t in np.arange(T)[::-1]:
            #print(s[t].T)
            dLdV += np.outer(delta_o[t], s[t].T)
            # Initial delta calculation: dL/dz
            #delta_t = self.V_NP.T.dot(delta_o[t]) * (1 - (s[t] ** 2))
            # Backpropagation through time (for at most self.bptt_truncate steps)
            #===================================================================
            # for bptt_step in np.arange(max(0, t-self.bptt_truncate), t+1)[::-1]:
            #     # print "Backpropagation step t=%d bptt step=%d " % (t, bptt_step)
            #     # Add to gradients at each previous step
            #     dLdW += np.outer(delta_t, s[bptt_step-1])              
            #     dLdU[:,x[bptt_step]] += delta_t
            #     # Update delta for next step dL/dz at t-1
            #     delta_t = self.W.T.dot(delta_t) * (1 - s[bptt_step-1] ** 2)
            #===================================================================
        return [dLdU, dLdV, dLdW]
    
    def predict(self, x):
        y = self.forward_propagation(x)[0]
        return y // 0.5
    
def gradient_check_theano(model, x, y, h=0.00001, error_threshold=0.01):
    # Overwrite the bptt attribute. We need to backpropagate all the way to get the correct gradient
    model.bptt_truncate = BPTT_TRUNCATE
    
    [y_out,s,sl, a] = model.forward_propagation(x)
    # Calculate the gradients using backprop
    bptt_gradients = model.bptt(x, y)
    
    bptt_gradients2 = model.bpttCalculate(x, y)
    # List of all parameters we want to chec.
    model_parameters = ['U', 'V',  'B',  'BO']
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
            yo = model.forward_propagation(X_train[i])
            #print(yo[1])
            bptt_gradients = model.bptt(X_train[i], y_train[i])
            o_error = model.sgd_step(X_train[i], y_train[i], learning_rate)
            print("output error")
            print(o_error)
            if(o_error[0] > 300 or o_error[0]<-300 or math.isnan(o_error[0])):
                print(o_error[0])
                gradient_check_theano(model, X_train[i], y_train[i])

            num_examples_seen += 1
        #save Parameter
        save_model_parameters_theano(MODEL_OUTPUT_FILE, model);
    
    
def save_model_parameters_theano(outfile, model):
    U, V, W = model.U.get_value(), model.V.get_value(), model.W.get_value()
    np.savez(outfile, U=U, V=V, W=W)
    #print ("Saved model parameters to %s." % outfile)
   
def load_model_parameters_theano(path, model):
    npzfile = np.load(path)
    U, V, W = npzfile["U"], npzfile["V"], npzfile["W"]
    model.hidden_dim = U.shape[0]
    model.word_dim = U.shape[1]
    model.U.set_value(U)
    model.V.set_value(V)
    model.W.set_value(W)
    #print ("Loaded model parameters from %s. hidden_dim=%d word_dim=%d" % (path, U.shape[0], U.shape[1]))
            
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
            