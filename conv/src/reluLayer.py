import numpy
from convMatrix import ConvMatrix
import math
import scipy
from scipy import signal

class ReluLayer(object):
    
    def forward(self, x):
        self.input = x
        self.out = ConvMatrix(self.input.d, self.input.y, self.input.x, self.input.params.clip(min=0), self.input.grads.copy())
        return self.out

    def backwards(self, y):
        self.input.grads[:] = self.out.grads
        zero_indexes = numpy.where(self.out.params < 0)
        self.input.grad()[zero_indexes] = 0

    def get_params_and_grads(self):
        return []

