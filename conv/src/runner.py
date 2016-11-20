import numpy
from convMatrix import ConvMatrix
import math
from convoNet import ConvNet
from convNetTrainer import ConvNetTrainer
from convLayer import ConvLayer
from fullyConnectedLayer import FullyConnectedLayer
from poolLayer import PoolLayer
from softMaxLayer import SoftmaxLayer
from reluLayer import ReluLayer
import os
from scipy.misc import toimage
import pickle

def load_CIFAR_batch(filename):
   with open(filename, 'rb') as f:
        datadict = pickle.load(f)
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        Y = numpy.array(Y)
        return X, Y

x, y = load_CIFAR_batch("../cifar10/data_batch_1")

l1 = ConvLayer(1, 2, 5, 5, 16)
l2 = ReluLayer()
l3 = PoolLayer(2, 2)
l4 = ConvLayer(1, 2, 5, 5, 20)
l5 = PoolLayer(2, 2)
l6 = ConvLayer(1, 2, 5, 5, 20)
l7 = PoolLayer(2, 2)
l8 = FullyConnectedLayer()
l9 = SoftmaxLayer()

layers = [l1, l2, l3, l4, l5, l6, l7, l8, l9]

convNet = ConvNet(layers)
trainer = ConvNetTrainer(0.0001, 0.95, 0.0000001, 4, convNet)

i = numpy.zeros((3, 32, 32))
i[0] = x[5][:, :, 0]
i[1] = x[5][:, :, 1]
i[2] = x[5][:, :, 2]

depth, y_input, x_input = i.shape
stats = trainer.train(ConvMatrix(depth, y_input, x_input, i), y)

