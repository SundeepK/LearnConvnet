import numpy
from convMatrix import ConvMatrix
import math
import scipy
from scipy import signal

class ReluLayer(object):
    
    def forward(self, x):
        self.input = x
        z, y, x = self.input.params.shape
        self.out = ConvMatrix(z, y, x, self.input.params.clip(min=0), self.input.grads.copy())
        return self.out

    def backwards(self, y):
        self.input.grads.fill(0)
        self.input.grads[:] = self.out.grads
        zero_indexes = numpy.where(self.out.params <= 0)
        self.input.grads[zero_indexes] = 0

    def get_params_and_grads(self):
        return []

    def get_bias_and_grads(self):
        return []

    def to_dict(self):
        return {
            'type': 'ReluLayer'
        }
