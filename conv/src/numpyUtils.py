import numpy
from convMatrix import ConvMatrix
import math
import scipy
from scipy import signal


def pad(input_matrix, padding):
    padded_input = numpy.pad(input_matrix.params.astype(float), pad_width=padding, mode='constant',
                             constant_values=0)
    if padded_input.shape[2] != input_matrix.params.shape[2]:
        padded_input = padded_input[padding:-padding, :, :]
    return padded_input


def im_2_col_indexes(input_m, filter_x, filter_y, stride):
    # flatten to single matrix
    input_z, input_y, input_x = input_m.shape
    total_size_for_1_dimension = input_x * input_y
    col_extent = input_x - filter_x + 1
    row_extent = input_y - filter_y + 1
    input_m = input_m.reshape(input_z * input_y, input_x)
    # Parameters
    input_y, input_x = input_m.shape
    # Get Starting block indices
    start_idx = (numpy.arange(filter_y)[:, None] * input_x + numpy.arange(filter_x))

    # Get off-setted indices across the height and width of input array
    offset_idx = (numpy.arange(0, row_extent, stride)[:, None] * input_x + numpy.arange(0, col_extent, stride))
    dimension_offsets = (numpy.arange(input_z).reshape(input_z, 1, 1) * (total_size_for_1_dimension))
    selected_indexes_as_col = numpy.hstack((start_idx.ravel()[:, None] + offset_idx.ravel()) + dimension_offsets)
    return selected_indexes_as_col

