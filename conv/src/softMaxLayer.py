import numpy
from matrix import Matrix
import math

class SoftmaxLayer(object):

    def forward(inputs):
        out_softmax = numpy.empty(len(inputs), dtype=object)
        for f in range(0, len(inputs)):
            layer = inputs[f]
            e_x = numpy.exp(layer - numpy.max(layer))
            out_softmax[f] = e_x / e_x.sum()

        return out_softmax
