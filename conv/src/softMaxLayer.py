import numpy
from convMatrix import ConvMatrix
import math

class SoftmaxLayer(object):

    def forward(self, inputs):
        self.inputs = inputs
        out_softmax = numpy.empty(inputs.x, dtype=object)
        max = numpy.max(inputs.params)
        for f in range(0, inputs.x):
            layer = inputs.params[f]
            e_x = numpy.exp((layer - max).astype(float))
            out_softmax[f] = e_x / e_x.sum()
        self.out = out_softmax
        print(out_softmax)
        return out_softmax

    def backwards(self, y):
        dws = numpy.zeros(self.inputs.x)
        dws[y] = 1
        dws = -(dws - self.out)
        self.dws = dws
        return dws

    def loss(self, y):
        if self.dws:
            return -(numpy.log(self.out))
        else:
            self.backwards(y)
            return -(numpy.log(self.out))

    def get_params_and_grads(self):
        return []
