import sys
import os
import operator
import sys
import math

def writeFile(filename, stringdata, type='a'):
    flog = open(filename, type)
    flog.write(stringdata)
    flog.close()