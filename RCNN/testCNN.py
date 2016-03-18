from theano.tensor.nnet import conv2d
import theano.tensor as T
import numpy as np
from ConvPoolLayer import ConvPoolLayer
from theano import theano
import math

a = np.arange(0,10,0.1)
b = np.arange(5,20,0.1)
x0 = np.array([  [   [a[i] * 0.1 + np.random.sample()/ 100.0, (b[i]) * 0.1 - np.random.sample()/ 100.0] for i in range(100)  ]   for j in range(2) ])
y0 = np.array([  [ [x0[0][i][0] //0.5 ] for i in range(100)  ] for j in range(2) ])

x1 = np.array([ a[i] * 0.1 + np.random.sample()/ 100.0 for i in range(100)  ])

window = 24
lenthX = int(len(x1) - window)

x = T.matrix('x')
layer0_input = x.reshape((1, 1, 1, window))

rng = np.random.RandomState(23455)



layer0 = ConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(1, 1, 1, window),
        filter_shape=(1, 1, 1, 5),
        poolsize=(1, 2)
    )



#prepare data
xi = []
for i in range(len(x1)):
    starti = i - window + 1
    e = [0 for col in range(-starti)]
    e.extend(x1[max(0,starti) : i + 1])
    #print(len(e))
    xi.append(e)
xinumpy = np.array(xi)
xis = theano.shared(name='xi', value=xinumpy.astype(theano.config.floatX))

index = T.iscalar()
action = theano.function([index],
                         [layer0.conv_out],
                         givens={
            x: xis[index: (index + 1)],
        })

learning_rate = 0.01
grads = T.grad(cost, layer0.params)
updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(layer0.params, grads)
    ]

v = [action(i) for i in range(lenthX)]
print(x1)
print(v)
print("end")