import numpy
from matrix import Matrix
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
                self.filters.append(Matrix(1, self.input_x, self.input_y))
        else:
            self.filters = filters
        if bias is None:
            self.bias = Matrix(self.depth, 1, 1)
        else:
            self.bias = bias


    def forward(self, input):
        for i in range(0, len(self.input)):
            a = numpy.sum(input[i] * self.filters[i].m)
            a = a + self.bias.m[i]
