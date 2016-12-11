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
import cifarUtils
from PIL import Image, ImageFilter
import io

class ConvNNRunner(object):

    def __init__(self, training_hook):
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
        self.layers = [l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12]
        self.convNet = ConvNet(self.layers)
        self.trainer = ConvNetTrainer(0.0001, 0.95, 0.00000001, 4, self.convNet)
        self.training_hook = training_hook

    def run_batch(self, batch):
        x, y = cifarUtils.load_CIFAR_batch(batch)
        i = numpy.zeros((3, 32, 32))
        for index in range(0, 3000):
            i[0] = x[index][:, :, 0]
            i[1] = x[index][:, :, 1]
            i[2] = x[index][:, :, 2]
            i = i /255.0-0.5
            depth, y_input, x_input = i.shape
            stats = self.trainer.train(ConvMatrix(depth, y_input, x_input, i.copy()), y[index])
            self.training_hook.onForwardProp(toimage(i), stats)

    def start(self):
        self.run_batch("./cifar10/data_batch_1")
        # self.run_batch("../cifar10/data_batch_2")
        # self.run_batch("../cifar10/data_batch_3")
        # self.run_batch("../cifar10/data_batch_4")
        # self.run_batch("../cifar10/data_batch_5")
