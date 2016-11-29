import numpy
from convMatrix import ConvMatrix
import math
import scipy
from scipy import signal

class InputLayer(object):

    def forward(self, x):
        self.input = x
        self.out = x
        return self.out

    def backwards(self, y):
        pass

    def get_params_and_grads(self):
        return []

    def get_bias_and_grads(self):
        return []

