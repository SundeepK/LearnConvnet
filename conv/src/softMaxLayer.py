import numpy
from convMatrix import ConvMatrix
import math


class SoftmaxLayer(object):

    def forward(self, inputs):
        self.inputs = inputs
        max = numpy.max(inputs.params)
        e_x = numpy.exp((inputs.params - max).astype(float))
        self.out = e_x / e_x.sum()
        self.es = e_x / e_x.sum()
        print(self.out)
        return self.out

    def backwards(self, y):
        self.inputs.grads.fill(0)
        self.inputs.grads[y] = 1
        self.inputs.grads[:] = -(self.inputs.grads - self.es)
        return -(numpy.log(self.es[y]))

    def get_params_and_grads(self):
        return []

    def get_bias_and_grads(self):
        return []

    def to_dict(self):
        return {
            'type': 'SoftmaxLayer'
        }
