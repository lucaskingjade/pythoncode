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

def saveFinalData(filename, data):
    f = open(filename, "w")
    for line in data:
        f.write(' '.join(line) + '\n')
    f.close()

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
        object = [start, end, audioType] = in_line.split(" ")
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
        object = [start, end, type, trajectory, number, complex] = in_line.split(" ")
        data.append(object);
    in_file.close()
    return head, data




def test():
    data = readAcoustics("2601/AA002601_brian.txt")
    head = ['time', 'pitch', 'RMS', 'intensity', 'loudness', 'noLaugh', 'smallLaugh', 'bigLaugh', 'laughingAndSpeaking', 'noGeste', 'prep', 'stroke', 'retract']
    head2, dataAudio = readAudioLabels("2601/AA002601_audio.txt")
    head3, dataPhase = readPhaseLabels("2601/AA002601_phase.txt")
    finalData = []
    finalData.append(head)
    
    startAudio = float(dataAudio[0][0])
    endAudio = float(dataAudio[0][1])
    startPhase = float(dataPhase[0][0])
    endPhase = float(dataPhase[0][1])
    
    
    for linedata in data:
        linedata
        currentTime = float(linedata[0])
        
        if(currentTime > endAudio and startAudio >= 0):
            del dataAudio[0]
            if(len(dataAudio) != 0):
                startAudio = float(dataAudio[0][0])
                endAudio = float(dataAudio[0][1])
            else:
                startAudio = -1
        
        if(currentTime > endPhase and startPhase >= 0):
            del dataPhase[0]
            if(len(dataPhase) != 0):
                startPhase = float(dataPhase[0][0])
                endPhase = float(dataPhase[0][1])
            else:
                startPhase = -1
            
        if(currentTime >= startAudio and currentTime <= endAudio and startAudio >= 0):
            label = dataAudio[0][2]
            if label == 'noLaugh':
                linedata.extend('1')
            else:
                linedata.extend('0')
                
            if label == 'smallLaugh':
                linedata.extend('1')
            else:
                linedata.extend('0')
                
            if label == 'bigLaugh':
                linedata.extend('1')
            else:
                linedata.extend('0')
                
            if label == 'laughingAndSpeaking':
                linedata.extend('1')
            else:
                linedata.extend('0')
        else:
            linedata.extend('1')
            linedata.extend('0')
            linedata.extend('0')
            linedata.extend('0')
                
                
            
        if(currentTime >= startPhase and currentTime <= endPhase and startPhase >= 0):
            label = dataPhase[0][2]
            if label == 'noGeste':
                linedata.extend('1')
            else:
                linedata.extend('0')
                
            if label == 'prep':
                linedata.extend('1')
            else:
                linedata.extend('0')
                
            if label == 'stroke':
                linedata.extend('1')
            else:
                linedata.extend('0')
                
            if label == 'retract':
                linedata.extend('1')
            else:
                linedata.extend('0')
        else:
            linedata.extend('1')
            linedata.extend('0')
            linedata.extend('0')
            linedata.extend('0')
        
        finalData.append(linedata)
        print(linedata)
    
    saveFinalData('AA002601_brianFinal.txt', finalData)
test()