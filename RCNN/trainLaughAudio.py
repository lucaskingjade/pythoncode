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
timeend = datetime.now()
print("data loading: %s second" %  (timeend - timestart).total_seconds())

#orgnizeddatainput, orgnizeddataoutput = prepareData(x, y, 2, 4)

#model = RNN(4, 10, 1)