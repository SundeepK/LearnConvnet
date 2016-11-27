import unittest
import numpy as np
from convMatrix import ConvMatrix
from convLayer import ConvLayer
import timeit
import time

class ConvLayerTest(unittest.TestCase):

    def test_conv_matrix(self):
        input = np.array([[[ 1.,  1.,  0.],
                           [ 2.,  2.,  0.],
                           [ 1.,  1.,  0.]],

                          [[ 1.,  1.,  0.],
                           [ 1.,  0.,  0.],
                           [ 0.,  1.,  0.]],

                          [[ 2.,  1.,  0.],
                           [ 0.,  1.,  0.],
                           [ 2.,  0.,  0.]]])

        underTest = ConvMatrix(3, 3, 3, input)
        assert np.array_equiv(underTest.params, input)
        assert np.array_equiv(underTest.grads, np.zeros((3, 3, 3)))



if __name__ == '__main__':
    unittest.main()

