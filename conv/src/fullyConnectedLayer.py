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
            bias = numpy.zeros(self.depth, dtype=float)
            bias.fill(0.1)
            self.bias = ConvMatrix(self.depth, 1, 1, bias.copy(), bias.copy())
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
        self.out = ConvMatrix(1, 1, self.depth, out, numpy.zeros(int(self.depth), dtype=float))
        return self.out

    def backwards(self, y):
        self.input.grads.fill(0)
        for i in range(0, self.depth):
            dw = self.out.grads[i]
            self.input.grads[:] = self.input.grads + (self.filters[i].params * dw)
            self.filters[i].grads[:] = self.filters[i].grads + (self.input.params * dw)
            self.bias.grads[:] = self.bias.grads + dw

    def get_input_and_grad(self):
        return self.input

    def get_params_and_grads(self):
        return self.filters

    def get_bias_and_grads(self):
        return [self.bias]

    def set_up_filters(self, z, y, x):
        if len(self.filters) <= 0:
            for i in range(0, self.depth):
                self.filters.append(ConvMatrix(z, y, x))

    def to_dict(self):
        out_filters = []
        for filter in self.filters:
            out_filters.append(filter.to_dict())
        return {
            'type': 'FullyConnectedLayer',
            'depth': self.depth,
            'filter_x': self.filter_x,
            'filter_y': self.filter_y,
            'bias': self.bias.to_dict(),
            'filters': out_filters
        }
