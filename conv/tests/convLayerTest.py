import unittest
import numpy as np
from matrix import Matrix
from convLayer import ConvLayer

class ConvLayerTest(unittest.TestCase):

    def test_forward_single_filter(self):
        input = np.array([[[ 1.,  1.,  0.],
                           [ 2.,  2.,  0.],
                           [ 1.,  1.,  0.]],

                          [[ 1.,  1.,  0.],
                           [ 1.,  0.,  0.],
                           [ 0.,  1.,  0.]],

                          [[ 2.,  1.,  0.],
                           [ 0.,  1.,  0.],
                           [ 2.,  0.,  0.]]])

        filter = np.array([[[-1.,  0.,  1.],
                            [-1., -1.,  1.],
                            [ 0.,  1.,  0.]],

                           [[ 1.,  0.,  1.],
                            [-1., -1., -1.],
                            [ 0.,  0., -1.]],

                           [[-1.,  0., -1.],
                            [ 1.,  1., -1.],
                            [ 0.,  1.,  1.]]])
        undertest = ConvLayer.with_filters([Matrix.with_matrix(filter)], 3, 3, 1, 0)
        array = np.zeros((1, 1, 1))
        array[0, 0, 0] = -5
        assert np.array_equiv(undertest.forward(input).m, array)

    def test_forward_multiple_filter_passes(self):
        input = np.array([[[ 0, 0, 1, 2, 1],
                           [ 2, 1, 0, 0, 0],
                           [ 0, 0, 2, 2, 0],
                           [ 0, 0, 0, 0, 0],
                           [ 1, 2, 0, 1, 1]],

                          [[ 1, 1, 0, 0, 0],
                           [ 2, 1, 1, 1, 1],
                           [ 0, 2, 1, 1, 1],
                           [ 1, 1, 2, 1, 1],
                           [ 0, 2, 0, 2, 1]],

                          [[ 1, 2, 0, 2, 0],
                           [ 0, 2, 2, 1, 0],
                           [ 0, 0, 1, 0, 0],
                           [ 0, 1, 0, 0, 1],
                           [ 0, 0, 0, 0, 2]]])

        filter = np.array([[[-1, 0, 1],
                            [1, 0, 0],
                            [1, -1, 1]],

                           [[-1, 1, 1],
                            [1, 1, 1],
                            [-1, 1, 0]],

                           [[1, -1, -1],
                            [ 0, 1, -1],
                            [ -1, -1, -1]]])
        undertest = ConvLayer.with_filters([Matrix.with_matrix(filter)], 5, 5, 2, 1)
        expected_filter_map = np.array([[[0, -5, 1],
                                         [4, 4, 4],
                                         [3, 9, 5]]])
        assert np.array_equiv(undertest.forward(input).m, expected_filter_map)


    def test_vectorized_forward_single_filter(self):
        input = np.array([[[ 1.,  1.,  0.],
                           [ 2.,  2.,  0.],
                           [ 1.,  1.,  0.]],

                          [[ 1.,  1.,  0.],
                           [ 1.,  0.,  0.],
                           [ 0.,  1.,  0.]],

                          [[ 2.,  1.,  0.],
                           [ 0.,  1.,  0.],
                           [ 2.,  0.,  0.]]])

        filter = np.array([[[-1.,  0.,  1.],
                            [-1., -1.,  1.],
                            [ 0.,  1.,  0.]],

                           [[ 1.,  0.,  1.],
                            [-1., -1., -1.],
                            [ 0.,  0., -1.]],

                           [[-1.,  0., -1.],
                            [ 1.,  1., -1.],
                            [ 0.,  1.,  1.]]])
        undertest = ConvLayer.with_filters([Matrix.with_matrix(filter)], 3, 3, 1, 0)
        array = np.zeros((1, 1, 1))
        array[0, 0, 0] = -5
        assert np.array_equiv(undertest.vectorized_forward(input).m, array)

    def test_vectorized_forward_multiple_filter_passes(self):
        input = np.array([[[ 0, 0, 1, 2, 1],
                           [ 2, 1, 0, 0, 0],
                           [ 0, 0, 2, 2, 0],
                           [ 0, 0, 0, 0, 0],
                           [ 1, 2, 0, 1, 1]],

                          [[ 1, 1, 0, 0, 0],
                           [ 2, 1, 1, 1, 1],
                           [ 0, 2, 1, 1, 1],
                           [ 1, 1, 2, 1, 1],
                           [ 0, 2, 0, 2, 1]],

                          [[ 1, 2, 0, 2, 0],
                           [ 0, 2, 2, 1, 0],
                           [ 0, 0, 1, 0, 0],
                           [ 0, 1, 0, 0, 1],
                           [ 0, 0, 0, 0, 2]]])

        filter = np.array([[[-1, 0, 1],
                            [1, 0, 0],
                            [1, -1, 1]],

                           [[-1, 1, 1],
                            [1, 1, 1],
                            [-1, 1, 0]],

                           [[1, -1, -1],
                            [ 0, 1, -1],
                            [ -1, -1, -1]]])

        np.array([[[0,  0,  0,  0, -1,  0,  0,  0,  0,  0,  0,  0,  0, -1, -1,  0, -1, -1,  0,  0,  0,  0,  2,  1, 0,  1,  0],
                  [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  2,  1,  1,  1,  2,  1,  0,  0,  0,  0, -2,  0, 0,  0, -1],
                  [ 0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  0,  1,  1,  0,  0,  0,  0, -2, -1,  0, 1,  0,  0],
                  [ 0,  0,  2,  0,  0,  2,  0,  2,  1,  0,  1,  0,  0,  2,  1,  0,  2,  2,  0,  0,  0,  0,  0,  0, 0,  0,  0],
                  [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  1,  1,  0,  0,  1,  1,  0,  0,  0,  1,  0, 0,  0,  2],
                  [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  2,  1,  0,  2,  2,  0, -2, -2,  0,  0,  0,  0, 0,  0,  0],
                  [ 0,  1,  0,  0,  0,  0,  0,  0,  0,  0, -1, -1,  0, -1, -1,  0,  0,  0,  0, -2, -1,  0, -1,  0, 0,  0,  0],
                  [-2,  0,  0,  0,  0,  0,  0,  0,  0,  2,  1,  1,  1,  2,  1,  0,  0,  0,  0, -2,  0,  0,  0, -1, 0,  0,  0],
                  [ 1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -2, -1,  0, -1,  0,  0, 0,  0,  0]]])

        undertest = ConvLayer.with_filters([Matrix.with_matrix(filter)], 5, 5, 2, 1)
        expected_filter_map = np.array([[[0, -5, 1],
                                         [4, 4, 4],
                                         [3, 9, 5]]])
        assert np.array_equiv(undertest.vectorized_forward(input).m, expected_filter_map)

if __name__ == '__main__':
    unittest.main()

