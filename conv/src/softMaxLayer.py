import numpy
from convMatrix import ConvMatrix
import math

class SoftmaxLayer(object):

    def forward(self, inputs):
        self.inputs = inputs
        max = numpy.max(inputs.params)
        e_x = numpy.exp((inputs.params - max).astype(float))
        self.out = e_x / e_x.sum()
        print(self.out)
        return self.out

    def backwards(self, y):
        self.inputs.grads = numpy.zeros(self.inputs.x)
        self.inputs.grads[y] = 1
        self.inputs.grads = -(self.inputs.grads - self.out)
        return -(numpy.log(self.out[y]))

    def get_params_and_grads(self):
        return []
