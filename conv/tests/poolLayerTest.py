import unittest
import numpy as np
from convMatrix import ConvMatrix
from poolLayer import PoolLayer
import timeit
import time

class PoolLayerTest(unittest.TestCase):

    def test_vectorized_forward_and_backprop_single_filter(self):
        # given
        input = np.array([[[1, 1, 2, 4],
                           [5, 6, 7, 8],
                           [3, 2, 1, 0],
                           [1, 2, 3, 4]],
                          [[1, 1, 2, 4],
                           [5, 1, 7, 8],
                           [3, 2, 1, 0],
                           [1, 2, 3, 4]]])

        input = ConvMatrix(2, 4, 4, input)

        expected = np.array([[[6, 8],
                              [3, 4]],
                             [[5, 8],
                              [3, 4]]])

        grad = np.array([[[2, 2],
                          [2, 2]],
                         [[2, 2],
                          [2, 2]]])

        undertest = PoolLayer(2, 2)

        # when
        out = undertest.forward(input)

        # then
        assert np.array_equiv(out.params, expected)

        # given
        out.grads[:] = grad

        # when
        undertest.backwards(0)

        # then
        expected_input_after_back_prop = np.array([[[0, 0, 0, 0],
                                                    [0, 2, 0, 2],
                                                    [2, 0, 0, 0],
                                                    [0, 0, 0, 2]],
                                                   [[0, 0, 0, 0],
                                                    [2, 0, 0, 2],
                                                    [2, 0, 0, 0],
                                                    [0, 0, 0, 2]]])

        assert np.array_equiv(input.grads, expected_input_after_back_prop)
