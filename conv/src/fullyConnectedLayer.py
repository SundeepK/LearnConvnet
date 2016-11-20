import numpy
from convMatrix import ConvMatrix
import math

class FullyConnectedLayer(object):

    def __init__(self, d=10, filters=None, bias=None):
        self.depth = d
        self.filter_x = 1
        self.filter_y = 1
        if filters is None:
            self.filters = []
        else:
            self.filters = filters
        if bias is None:
            self.bias = ConvMatrix(1, 1, self.depth, numpy.empty(self.depth))
        else:
            self.bias = bias

    def forward(self, input):
        self.input = input
        z, y, x = self.input.params.shape
        self.set_up_filters(z, y, x)
        out = numpy.empty(self.depth)
        for i in range(0, self.depth):
            a = numpy.sum(input.params * self.filters[i].params)
            out[i] = a + self.bias.params[i]
        self.out = ConvMatrix(1, 1, self.depth, out)
        return self.out

    def backwards(self, y):
        self.input.grad().fill(0)
        for i in range(0, self.depth):
            dw = self.out.grad()[i]
            self.input.grad = self.input.grad + (self.filters[i].params * dw)
            self.filters[i].grad = self.filters[i].grad + (self.input.params * dw)
            self.bias.grad = self.bias.params + dw

    def get_input_and_grad(self):
        return self.input

    def get_params_and_grads(self):
        return self.filters

    def set_up_filters(self, z, y, x):
        if len(self.filters) <= 0:
            for i in range(0, self.depth):
                self.filters.append(ConvMatrix(z, y, x))
