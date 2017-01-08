import numpy
from math import sqrt


class ConvMatrix(object):

    def __init__(self, d, y, x, matrix=None, grad_matrix=None):
        self.d = d
        self.y = y
        self.x = x
        if matrix is None:
            sigma = sqrt(1.0/(x*y*d))
            self.params = numpy.random.normal(0, sigma, (d, y, x))
        else:
            self.params = matrix
        if grad_matrix is None:
            self.grads = numpy.zeros((int(d), int(y), int(x)), dtype=float)
        else:
            self.grads = grad_matrix

    @classmethod
    def with_matrix(cls, matrix):
        obj = cls(matrix.shape[0], matrix.shape[1], matrix.shape[2], matrix, None)
        return obj

    def x(self):
        return self.x

    def y(self):
        return self.y

    def d(self):
        return self.d

    def set_grad(self, g):
        self.grads[:] = g

    def set_params(self, p):
        self.params[:] = p

    def to_dict(self):
        return {
            'd': self.d,
            'y': self.y,
            'x': self.x,
            'params': self.params.tolist(),
            'grads': self.grads.tolist()
        }
