import numpy
from math import sqrt


class Matrix(object):

    def __init__(self, d, x, y, matrix=None):
        self.x = x
        self.y = y
        self.d = d
        if matrix is None:
            sigma = sqrt(1.0/(x*y*d))
            self.m = numpy.random.normal(0, sigma, (d, x, y))
        else:
            self.m = matrix

    @classmethod
    def with_matrix(cls, matrix):
        obj = cls(matrix.shape[0], matrix.shape[1], matrix.shape[2], matrix)
        return obj

    def m(self):
        return self.m

    def x(self):
        return self.x

    def y(self):
        return self.y
