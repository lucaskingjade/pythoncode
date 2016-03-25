import numpy as np

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


def test():
    a = np.arange(0,10,0.1)
    b = np.arange(5,20,0.1)
    bias = [np.random.sample()/ 100.0 for i in range(100)]
    bias2 = [np.random.sample()/ 100.0 for i in range(100)]
    bias3 = [np.random.sample()/ 100.0 for i in range(100)]
    x = np.array([  [   [a[i] * 0.1 + np.random.sample()/ 100.0, (b[i]) * 0.1 - np.random.sample()/ 100.0,] for i in range(100)]   for j in range(2) ])
    y = np.array([  [ [x[0][i][0] //0.5 ] for i in range(100)  ] for j in range(2) ])

    orgnizeddatainput, orgnizeddataoutput = prepareData(x, y, 2, 2)

test()