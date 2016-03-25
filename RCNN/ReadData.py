import sys
import os
import operator
import sys
import math
from enum import Enum

class AcousticsIndex(Enum):
    time = 0
    pitch = 1
    RMS = 2
    intensity = 3
    loudness = 4

def readAcoustics(filename):
    data = []
    in_file = open(filename, "rt")
    while True:
        in_line = in_file.readline()
        if not in_line:
            break
        in_line = in_line[:-1]
        object = [time, pitch, RMS, intensity, loudness] = in_line.split(" ")
        data.append(object);
    in_file.close()
    return data

def readAudioLabels(filename):
    data = []
    in_file = open(filename, "rt")
    in_line = in_file.readline()
    head = in_line.split(" ")
    while True:
        in_line = in_file.readline()
        if not in_line:
            break
        in_line = in_line[:-1]
        object = [start, end , audioType] = in_line.split(" ")
        data.append(object);
    in_file.close()
    return head, data

def readBreathLabels(filename):
    data = []
    in_file = open(filename, "rt")
    in_line = in_file.readline()
    head = in_line.split(" ")
    while True:
        in_line = in_file.readline()
        if not in_line:
            break
        in_line = in_line[:-1]
        object = [start, end , breathType] = in_line.split(" ")
        data.append(object);
    in_file.close()
    return head, data
    
def readPhaseLabels(filename):
    data = []
    in_file = open(filename, "rt")
    in_line = in_file.readline()
    head = in_line.split(" ")
    while True:
        in_line = in_file.readline()
        if not in_line:
            break
        in_line = in_line[:-1]
        object = in_line.split(" ")
        data.append(object);
    in_file.close()
    return head, data


def test():
    #data = readAcoustics("C:/Users/Jing/Videos/2601/AA002601_brian.txt")
    #head2, data2 = readAudioLabels("C:/Users/Jing/Videos/2601/AA002601_audio.txt")
    head3, data3 = readBreathLabels("C:/Users/Jing/Videos/2601/AA002601_breath.txt")
    head4, data4 = readBreathLabels("C:/Users/Jing/Videos/2601/AA002601_phase.txt")
    
test()