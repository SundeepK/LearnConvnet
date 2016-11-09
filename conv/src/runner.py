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
import pickle
from scipy.misc import toimage

def load_CIFAR_batch(filename):
   with open(filename, 'rb') as f:
        datadict = pickle.load(f)
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        Y = numpy.array(Y)
        return X, Y

x, y = load_CIFAR_batch("../cifar10/data_batch_1")

print(x[0].shape)
depth, y_input, x_input = x[0].shape


l1 = ConvLayer(1, 2, 5, 5, 16, depth)
l2 = ReluLayer()
l3 = PoolLayer(2, 2)
l4 = ConvLayer(1, 2, 5, 5, 20, depth)
l5 = PoolLayer(2, 2)
l6 = ConvLayer(1, 2, 5, 5, 20, depth)
l7 = PoolLayer(2, 2)
l8 = SoftmaxLayer()

layers = [l1, l2, l3, l4, l5, l6, l7, l8]

convNet = ConvNet(layers)
trainer = ConvNetTrainer(0.0001, 0.95, 0.0000001, 4, convNet)
# stats = trainer.train(ConvMatrix(depth, y_input, x_input, x[0].transpose().transpose()), y)
# print(stats.cost_loss)

