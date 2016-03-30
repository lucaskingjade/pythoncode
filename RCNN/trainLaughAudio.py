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

npx, npy= buildData('laugh')
print('data is load')