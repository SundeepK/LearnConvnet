import unittest
import numpy as np
from convMatrix import ConvMatrix
from poolLayer import PoolLayer
import timeit
import time

class PoolLayerTest(unittest.TestCase):

    def test_vectorized_forward_single_filter(self):
        input = np.array([[[1, 1, 2, 4],
                           [5, 6, 7, 8],
                           [3, 2, 1, 0],
                           [1, 2, 3, 4]],
                          [[1, 1, 2, 4],
                           [5, 1, 7, 8],
                           [3, 2, 1, 0],
                           [1, 2, 3, 4]]])

        expected = np.array([[[6, 8],
                              [3, 4]],
                             [[5, 8],
                              [3, 4]]])

        undertest = PoolLayer(2, 2)
        assert np.array_equiv(undertest.forward(ConvMatrix(1, 4, 4, input)).params, ConvMatrix(1, 2, 2, expected).params)
