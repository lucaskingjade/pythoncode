import sys
import os
import operator
import sys
import math
from enum import Enum
import numpy as np

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




def translateData(file1, file2, file3):
    #head = ['time', 'pitch', 'RMS', 'intensity', 'loudness', 'noLaugh', 'smallLaugh', 'bigLaugh', 'laughingAndSpeaking', 'noGeste', 'prep', 'stroke', 'retract']
    head = ['time', 'pitch', 'RMS', 'intensity', 'loudness', 'no', 'speaking', 'laughingAndSpeaking', 'smallLaugh', 'bigLaugh']
    #data = readAcoustics("2601/AA002601_brian.txt")
    #head2, dataAudio = readAudioLabels("2601/AA002601_audio.txt")
    #head3, dataPhase = readPhaseLabels("2601/AA002601_phase.txt")
    data = readAcoustics(file1+'.txt')
    head2, dataAudio = readAudioLabels(file2+'.txt')
    #head3, dataPhase = readPhaseLabels(file3+'.txt')
    finalData = []
    finalData.append(head)
    
    startAudio = float(dataAudio[0][0])
    endAudio = float(dataAudio[0][1])
    #startPhase = float(dataPhase[0][0])
    #endPhase = float(dataPhase[0][1])
    
    
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
        
#         if(currentTime > endPhase and startPhase >= 0):
#             del dataPhase[0]
#             if(len(dataPhase) != 0):
#                 startPhase = float(dataPhase[0][0])
#                 endPhase = float(dataPhase[0][1])
#             else:
#                 startPhase = -1
            
        if(currentTime >= startAudio and currentTime <= endAudio and startAudio >= 0):
            label = dataAudio[0][2]
            if label == 'no':
                linedata.extend('1')
            else:
                linedata.extend('0')
            if label == 'speaking':
                linedata.extend('1')
            else:
                linedata.extend('0')
            
            if label == 'laughingAndSpeaking':
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
                
            
        else:
            linedata.extend('1')
            linedata.extend('0')
            linedata.extend('0')
            linedata.extend('0')
            linedata.extend('0')
                
                
            
#         if(currentTime >= startPhase and currentTime <= endPhase and startPhase >= 0):
#             label = dataPhase[0][2]
#             if label == 'noGeste':
#                 linedata.extend('1')
#             else:
#                 linedata.extend('0')
#                 
#             if label == 'prep':
#                 linedata.extend('1')
#             else:
#                 linedata.extend('0')
#                 
#             if label == 'stroke':
#                 linedata.extend('1')
#             else:
#                 linedata.extend('0')
#                 
#             if label == 'retract':
#                 linedata.extend('1')
#             else:
#                 linedata.extend('0')
#         else:
#             linedata.extend('1')
#             linedata.extend('0')
#             linedata.extend('0')
#             linedata.extend('0')
        
        finalData.append(linedata)
        print(linedata)
    
    saveFinalData(file1+'Final.txt', finalData)
    
def readFinalData(filename):
    data = []
    in_file = open(filename, "rt")
    in_line = in_file.readline()
    head = in_line.split(" ")
    while True:
        in_line = in_file.readline()
        if not in_line:
            break
        in_line = in_line[:-1]
        #object = [time, pitch, RMS, intensity, loudness, noLaugh, smallLaugh, bigLaugh, laughingAndSpeaking, noGeste, prep, stroke, retract] = in_line.split(" ")
        object = [time, pitch, RMS, intensity, loudness, no, spearking, laughingAndSpeaking,  smallLaugh, bigLaugh] = in_line.split(" ")
        data.append(object);
    in_file.close()
    return head, data

def getData(type, data):
    if(type == 'laugh'):
        object = [[float(data[i][j]) for j in range(10)] for i in range(len(data))]
        #print(objet[2164])
        return object
    elif (type == 'geste'):
        object = [[float(data[i][j]) for j in [0,1,2,3,4,9,10,11,12]] for i in range(len(data))]
        #print(objet[2164])
        return object
    
def getXY(data):
    a = [0.002, 1000, 0.01, 0.02]
    x = [[data[i][j] * a[j-1] for j in [1,2,3,4]] for i in range(len(data))]
    #y = [[data[i][j] for j in [5,6,7,8,9]] for i in range(len(data))]
    y = []
    for i in range(len(data)):
        if (data[i][5] == 1):
            y.append([1,0,0])
        elif (data[i][6] == 1):
            y.append([0,1,0])
        else:
            y.append([0,0,1])  
    y2 = [[data[i][j], 1 - data[i][j]] for j in [5] for i in range(len(data))]
    
    return x, y, y2

def buildData(datapath = 'C:/Users/Jing/Videos/LaughterAnnotation/new/', type='laugh'):
    print(type)
    #files = ['AA002501_brianFinal','AA002601_brianFinal','AA002602_brianFinal','AA000201_BriceFinal','AA000201_carolineFinal','AA000202_BriceFinal','AA000202_carolineFinal','AA000301_kenFinal','AA000302_kenFinal','AA000303_kenFinal']
    files = ['AA002501_brianFinal','AA002601_brianFinal','AA002602_brianFinal','AA000201_BriceFinal','AA000201_carolineFinal','AA000202_BriceFinal','AA000202_carolineFinal','AA000301_kenFinal']
    
    path = datapath
    datasetX = []
    datasetY = []
    datasetY2 = []
    for file in files:
        head, data = readFinalData(path+file+'.txt')
        d = getData(type, data)
        #d = getData('geste', data)
        x, y, y2 = getXY(d)
        datasetX.append(x)
        datasetY.append(y)
        datasetY2.append(y2)
    npX = np.asarray(datasetX) 
    npY = np.asarray(datasetY) 
    npY2 = np.asarray(datasetY2) 
    return npX, npY, npY2
        
        
#buildData()   
# translateData('C:/Users/Jing/Videos/all/AA000201_Brice','C:/Users/Jing/Videos/LaughterAnnotation/new/AA000201(1)_audio', 'C:/Users/Jing/Videos/LaughterAnnotation/AA000201(1)_phase') 
# translateData('C:/Users/Jing/Videos/all/AA000201_caroline','C:/Users/Jing/Videos/LaughterAnnotation/new/AA000201_audio', 'C:/Users/Jing/Videos/LaughterAnnotation/AA000201_phase') 
# translateData('C:/Users/Jing/Videos/all/AA000202_Brice','C:/Users/Jing/Videos/LaughterAnnotation/new/AA000202(1)_audio', 'C:/Users/Jing/Videos/LaughterAnnotation/AA000202(1)_phase') 
# translateData('C:/Users/Jing/Videos/all/AA000202_caroline','C:/Users/Jing/Videos/LaughterAnnotation/new/AA000202_audio', 'C:/Users/Jing/Videos/LaughterAnnotation/AA000202_phase') 
# 
# translateData('C:/Users/Jing/Videos/all/AA000301_ken','C:/Users/Jing/Videos/LaughterAnnotation/new/AA000301_audio', 'C:/Users/Jing/Videos/LaughterAnnotation/AA000301_phase') 
# #translateData('C:/Users/Jing/Videos/all/AA000302_ken','C:/Users/Jing/Videos/LaughterAnnotation/AA000302_audio', 'C:/Users/Jing/Videos/LaughterAnnotation/AA000302_phase') 
# #translateData('C:/Users/Jing/Videos/all/AA000303_ken','C:/Users/Jing/Videos/LaughterAnnotation/AA000303_audio', 'C:/Users/Jing/Videos/LaughterAnnotation/AA000303_phase') 
# 
# translateData('C:/Users/Jing/Videos/all/AA002501_brian','C:/Users/Jing/Videos/LaughterAnnotation/new/AA002501_audio', 'C:/Users/Jing/Videos/LaughterAnnotation/AA002501_phase') 
# translateData('C:/Users/Jing/Videos/all/AA002601_brian','C:/Users/Jing/Videos/LaughterAnnotation/new/AA002601_audio', 'C:/Users/Jing/Videos/LaughterAnnotation/AA002601_phase') 
# translateData('C:/Users/Jing/Videos/all/AA002602_brian','C:/Users/Jing/Videos/LaughterAnnotation/new/AA002602_audio', 'C:/Users/Jing/Videos/LaughterAnnotation/AA002602_phase') 

#buildData()

#head, data = readFinalData('AA002601_brianFinal.txt')
#data1 = getData('laugh', data)
#data2 = getData('geste', data)
#x, y = getXY(data1)
