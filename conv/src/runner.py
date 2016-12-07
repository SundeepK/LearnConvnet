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
from inputLayer import InputLayer
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

def run(trainer, batch):
     x, y = load_CIFAR_batch(batch)
     i = numpy.zeros((3, 32, 32))
     for index in range(0, 3000):
          print("Training no: " + str(index) + " batch: " + batch)
          i[0] = x[index][:, :, 0]
          i[1] = x[index][:, :, 1]
          i[2] = x[index][:, :, 2]
          i = i /255.0-0.5
          depth, y_input, x_input = i.shape
          stats = trainer.train(ConvMatrix(depth, y_input, x_input, i.copy()), y[index])
          # print(stats.cost_loss)

l1 = InputLayer()
l2 = ConvLayer(1, 2, 5, 5, 16, "l8")
l3 = ReluLayer()
l4 = PoolLayer(2, 2, "l4")
l5 = ConvLayer(1, 2, 5, 5, 20, "l8")
l6 = ReluLayer()
l7 = PoolLayer(2, 2, "l7")
l8 = ConvLayer(1, 2, 5, 5, 20, "l8")
l9 = ReluLayer()
l10 = PoolLayer(2, 2, "l10")
l11 = FullyConnectedLayer()
l12 = SoftmaxLayer()

layers = [l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12]

convNet = ConvNet(layers)
trainer = ConvNetTrainer(0.0001, 0.95, 0.00000001, 4, convNet)

run(trainer, "../cifar10/data_batch_1")
run(trainer, "../cifar10/data_batch_2")
run(trainer, "../cifar10/data_batch_3")
# run(trainer, "../cifar10/data_batch_4")
# run(trainer, "../cifar10/data_batch_5")



