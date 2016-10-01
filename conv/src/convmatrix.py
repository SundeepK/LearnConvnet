import numpy
from math import sqrt


class ConvMatrix(object):

    def __init__(self, d, x, y, matrix=None, grad_matrix=None):
        self.x = x
        self.y = y
        self.d = d
        if matrix is None:
            sigma = sqrt(1.0/(x*y*d))
            self.params = numpy.random.normal(0, sigma, (d, y, x))
        else:
            self.params = matrix
        if grad_matrix is None:
            self.grads = numpy.zeros((d, y, x))
        else:
            self.grads = grad_matrix

    @classmethod
    def with_matrix(cls, matrix):
        obj = cls(matrix.shape[0], matrix.shape[2], matrix.shape[1], matrix, None)
        return obj

    def params(self):
        return self.params

    def grad(self):
        return self.grads

    def x(self):
        return self.x

    def y(self):
        return self.y

    def d(self):
        return self.d
