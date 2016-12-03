import unittest
import numpy as np
from convMatrix import ConvMatrix
from convLayer import ConvLayer
import timeit
import time

class ConvLayerTest(unittest.TestCase):

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
        undertest = ConvLayer.with_filters([ConvMatrix.with_matrix(filter)], 1, 0)
        array = np.zeros((1, 1, 1))
        array[0, 0, 0] = -5
        assert np.array_equiv(undertest.forward(ConvMatrix(3, 3, 3, input)).params, array)

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

        undertest = ConvLayer.with_filters([ConvMatrix.with_matrix(filter)], 2, 1)
        expected_filter_map = np.array([[[0, -5, 1],
                                         [4, 4, 4],
                                         [3, 9, 5]]])
        start_time = timeit.default_timer()
        assert np.array_equiv(undertest.forward(ConvMatrix(3, 5, 5, input)).params, expected_filter_map)
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

        undertest = ConvLayer.with_filters([ConvMatrix.with_matrix(filter)], 2, 1)
        expected_filter_map = np.array([[[2, -2, 1],
                                         [3, 3, 4],
                                         [1, 6, 5]]])
        start_time = timeit.default_timer()
        assert np.array_equiv(undertest.forward(ConvMatrix(3, 5, 5, input)).params, expected_filter_map)
        print(timeit.default_timer() - start_time)

    def test_it_performs_backprop_multiple_filter_passes(self):
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


        undertest = ConvLayer.with_filters([ConvMatrix.with_matrix(filter)], 2, 1)
        expected_filter_map = np.array([[[0, -5, 1],
                                         [4, 4, 4],
                                         [3, 9, 5]]])

        out_result = undertest.forward(ConvMatrix(3, 5, 5, input))
        assert np.array_equiv(out_result.params, expected_filter_map)

        new_grad = np.array([[[0.5, 0.5, 0.5],
                              [0.4, 0.4, 0.4],
                              [0.3, 0.3, 0.3]]])
        out_result.grads[:] = new_grad

        undertest.backwards(1)

        expected_input_grad = np.array([[
                                      [ 0. ,  0.5,  0. ,  0.5,  0. ],
                                      [-0.5,  1. , -0.5,  1. , -0.5],
                                      [ 0. ,  0.4,  0. ,  0.4,  0. ],
                                      [-0.4,  0.8, -0.4,  0.8, -0.4],
                                      [ 0. ,  0.3,  0. ,  0.3,  0. ]],

                                     [[ 0.5,  1. ,  0.5,  1. ,  0.5],
                                      [ 0.9, -0.5,  0.9, -0.5,  0.9],
                                      [ 0.4,  0.8,  0.4,  0.8,  0.4],
                                      [ 0.7, -0.4,  0.7, -0.4,  0.7],
                                      [ 0.3,  0.6,  0.3,  0.6,  0.3]],

                                     [[ 0.5, -0.5,  0.5, -0.5,  0.5],
                                      [-0.9, -1. , -0.9, -1. , -0.9],
                                      [ 0.4, -0.4,  0.4, -0.4,  0.4],
                                      [-0.7, -0.8, -0.7, -0.8, -0.7],
                                      [ 0.3, -0.3,  0.3, -0.3,  0.3]]])

        expected_filter_grad = np.array([[[ 0.4,  0.8,  0.4],
                                          [ 2.7,  2.4,  2.7],
                                          [ 0.5,  1. ,  0.5]],

                                         [[ 1.4,  2.8,  1.4],
                                          [ 2.9,  1.6,  2.9],
                                          [ 1.8,  3.6,  1.8]],

                                         [[ 1.5,  1.1,  1.5],
                                          [ 2. ,  1.5,  2. ],
                                          [ 1.9,  1.4,  1.9]]])

        grads = undertest.get_input_and_grad().grads
        np.testing.assert_array_almost_equal_nulp(grads, expected_input_grad)
        np.testing.assert_array_almost_equal_nulp(undertest.get_params_and_grads()[0].grads, expected_filter_grad)

if __name__ == '__main__':
    unittest.main()

