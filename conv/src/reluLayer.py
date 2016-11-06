import numpy
from convMatrix import ConvMatrix
import math
import scipy
from scipy import signal

class ReluLayer(object):
    
    def __init__(self, out_sx, out_sy, out_depth):
        self.out_sx = out_sx
        self.out_sy = out_sy
        self.out_depth = out_depth

    def forward(self, x):
        self.input = x
        self.out = ConvMatrix(self.input.d, self.input.y, self.input.x, self.input.params.clip(min=0), self.input.grad().copy())
        return self.out

    def backwards(self):
        self.input.grad()[:] = self.out.grad()
        zero_indexes = numpy.where(self.out.params < 0)
        self.input.grad()[zero_indexes] = 0
