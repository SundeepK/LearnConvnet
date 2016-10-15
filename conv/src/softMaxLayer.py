import numpy
from convMatrix import ConvMatrix
import math

class SoftmaxLayer(object):

    def forward(self, inputs):
        self.inputs = inputs
        out_softmax = numpy.empty(len(inputs), dtype=object)
        max = numpy.max(inputs)
        for f in range(0, len(inputs)):
            layer = inputs[f]
            e_x = numpy.exp(layer - max)
            out_softmax[f] = e_x / e_x.sum()
        self.out = out_softmax
        return out_softmax

    def backward(self, y):
        dws = numpy.zeros(len(self.inputs))
        dws[y] = 1
        dws = -(dws - self.out)
        self.dws = dws
        return dws

    def loss(self, y):
        if self.dws:
            return -(numpy.log(self.out))
        else:
            self.backward(y)
            return -(numpy.log(self.out))
