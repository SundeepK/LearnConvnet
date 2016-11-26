import unittest
import numpy as np
from convMatrix import ConvMatrix
from reluLayer import ReluLayer
import timeit
import time

class ReluLayerTest(unittest.TestCase):

    def test_it_activate_when_greater_than_zero(self):
        input_matrix = np.array([[[ -1.,  1.,  1.],
                                   [ 1.,  1.,  1.],
                                   [ 1.,  1.,  1.]],

                                  [[ -2.,  1.,  1.],
                                   [ 1.,  1.,  1.],
                                   [ 1.,  1.,  1.]],

                                  [[ -3.,  1.,  1.],
                                   [ 1.,  1.,  1.],
                                   [ 1.,  1.,  1.]]])

        expected = np.array([[[ 0.,  1.,  1.],
                              [ 1.,  1.,  1.],
                              [ 1.,  1.,  1.]],

                             [[ 0.,  1.,  1.],
                              [ 1.,  1.,  1.],
                              [ 1.,  1.,  1.]],

                             [[ 0.,  1.,  1.],
                              [ 1.,  1.,  1.],
                              [ 1.,  1.,  1.]]])

        undertest = ReluLayer()
        input = ConvMatrix(3, 3, 3, input_matrix, None)
        out = undertest.forward(input)
        assert np.array_equiv(out.params, expected)
        out.grads[:] = 10
        out.params[0, 0, 2] = -1
        out.params[1, 0, 2] = -2
        out.params[2, 0, 2] = -3

        expected = np.array([[[ 10.,  10.,  0.],
                              [ 10.,  10.,  10.],
                              [ 10.,  10.,  10.]],

                             [[ 10.,  10.,  0.],
                              [ 10.,  10.,  10.],
                              [ 10.,  10.,  10.]],

                             [[ 10.,  10.,  0.],
                              [ 10.,  10.,  10.],
                              [ 10.,  10.,  10.]]])
        undertest.backwards(0)
        assert np.array_equiv(input.grads, expected)

if __name__ == '__main__':
    unittest.main()

