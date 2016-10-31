import numpy
from convMatrix import ConvMatrix
import math

class FullyConnectedLayer(object):

    def __init__(self, input_x, input_y, d, filters=None, bias=None):
        self.depth = d
        self.filter_x = 1
        self.filter_y = 1
        self.input_x = input_x
        self.input_y = input_y
        if filters is None:
            self.filters = []
            for i in range(0, d):
                self.filters.append(ConvMatrix(1, self.input_x, self.input_y))
        else:
            self.filters = filters
        if bias is None:
            self.bias = ConvMatrix(self.depth, 1, 1)
        else:
            self.bias = bias


    def forward(self, input):
        self.input = input
        self.out = numpy.empty(len(input), dtype=object)
        for i in range(0, len(self.input)):
            a = numpy.sum(input[i] * self.filters[i].params)
            self.out[i] = a + self.bias.params[i]
        return self.out

    def backward(self):
        for i in range(0, len(self.filters)):
            dw = self.out[i]
            self.input[i].grad = (self.filters[i].params * dw)
            self.filters[i].grad = self.input[i].params * dw
            self.bias.grad = self.bias.params + dw

    def get_input_and_grad(self):
        return self.input

    def get_params_and_grads(self):
        return self.filters
