import numpy
from convMatrix import ConvMatrix
import math

class SoftmaxLayer(object):

    def forward(self, inputs):
        self.inputs = inputs
        out_softmax = numpy.empty(inputs.d, dtype=object)
        max = numpy.max(inputs.params)
        for f in range(0, inputs.d):
            layer = inputs[f]
            e_x = numpy.exp(layer - max)
            out_softmax[f] = e_x / e_x.sum()
        self.out = out_softmax
        return out_softmax

    def backward(self, y):
        dws = numpy.zeros(self.inputs.d)
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

    def get_params_and_grads(self):
        return []
