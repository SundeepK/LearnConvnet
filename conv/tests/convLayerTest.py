import unittest
import numpy as np
from matrix import Matrix
from convLayer import ConvLayer
import timeit
import time

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
        start_time = timeit.default_timer()
        assert np.array_equiv(undertest.forward(input).m, expected_filter_map)
        print(timeit.default_timer() - start_time)



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
        assert np.array_equiv(undertest.vectorized_forward(input)[0], array)

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

        undertest = ConvLayer.with_filters([Matrix.with_matrix(filter)], 5, 5, 2, 1)
        expected_filter_map = np.array([[[0, -5, 1],
                                         [4, 4, 4],
                                         [3, 9, 5]]])
        start_time = timeit.default_timer()
        assert np.array_equiv(undertest.vectorized_forward(input)[0], expected_filter_map)
        print(timeit.default_timer() - start_time)

    def test_vectorized_forward_multiple_filter_passes_2(self):
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

        filter = np.array([[[-1, 0],
                            [1, 0],
                            [1, -1]],

                           [[-1, 1],
                            [1, 1],
                            [-1, 1]],

                           [[1, -1],
                            [ 0, 1],
                            [ -1, -1]]])

        out =  np.array([[ 0,  0,  0,  0, -1,  0,  0,  0,  0,  0,  0,  0,  0, -1, -1,  0, -1, -1,  0,  0,  0,  0,  2,  1, 0 , 1,  0],
                [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  2,  1,  1,  1,  2,  1,  0,  0,  0,  0, -2,  0, 0 , 0, -1],
                [ 0,  0,  2,  0,  0,  2,  0,  2,  1,  0,  1,  0,  0,  2,  1,  0,  2,  2,  0,  0,  0,  0,  0,  0, 0 , 0,  0],
                [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  1,  1,  0,  0,  1,  1,  0,  0,  0,  1,  0, 0 , 0,  2],
                [ 0,  1,  0,  0,  0,  0,  0,  0,  0,  0, -1, -1,  0, -1, -1,  0,  0,  0,  0, -2, -1,  0, -1,  0, 0 , 0,  0],
                [-2,  0,  0,  0,  0,  0,  0,  0,  0,  2,  1,  1,  1,  2,  1,  0,  0,  0,  0, -2,  0,  0,  0, -1, 0 , 0,  0]])

        undertest = ConvLayer.with_filters([Matrix.with_matrix(filter)], 5, 5, 2, 1)
        expected_filter_map = np.array([[[2, -2, 1],
                                         [3, 3, 4],
                                         [1, 6, 5]]])
        start_time = timeit.default_timer()
        assert np.array_equiv(undertest.vectorized_forward(input)[0], expected_filter_map)
        print(timeit.default_timer() - start_time)


if __name__ == '__main__':
    unittest.main()

