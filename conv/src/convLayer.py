import numpy
from convMatrix import ConvMatrix
import math
import scipy
from scipy import signal

class ConvLayer(object):

    def __init__(self, inputX, inputY, stride, padding, filter_x, filter_y, filter_d, d, filters=None):
        self.stride = stride
        self.padding = padding
        self.input_x = inputX
        self.input_y = inputY
        if filters is None:
            self.filters = []
            self.depth = d
            self.filter_x = filter_x
            self.filter_y = filter_y
            for i in range(0, d):
                self.filters.append(ConvMatrix(filter_d, filter_x, filter_y))
        else:
            self.depth = d
            self.filter_x = filter_x
            self.filter_y = filter_y
            self.filters = filters
        self.out_filter_map_x = int(math.floor((self.input_x - self.filter_x + (self.padding * 2)) / self.stride) + 1)
        self.out_filter_map_y = int(math.floor((self.input_y - self.filter_y + (self.padding * 2)) / self.stride) + 1)

    @classmethod
    def with_filters(cls, filters, input_x, input_y, stride, padding):
        obj = cls(input_x, input_y, stride, padding, filters[0].x, filters[0].y, filters[0].d, len(filters), filters)
        return obj

    def forward(self, input_conv):
        padded_input = numpy.pad(input_conv, pad_width=self.padding, mode='constant', constant_values=0)
        if padded_input.shape[2] != input_conv.shape[2]:
            padded_input = padded_input[1:-1, :, :]

        p_z, p_y, p_x = padded_input.shape
        self.input_conv = ConvMatrix(p_z, p_x, p_y, padded_input)
        self.input_2_col, offset_idx, self.input_rolled_out_indexes = self.im_2_col(padded_input)

        # Get all actual indices & index into input array for final output
        out_filter_map = ConvMatrix(self.depth, self.out_filter_map_x, self.out_filter_map_y, None, None)
        for f in range(0, len(self.filters)):
            f_z, f_y, f_x = self.filters[f].params.shape
            filter_reshaped = numpy.repeat(self.filters[f].params.reshape(f_z, 1, f_x * f_y),
                                           self.out_filter_map_y * self.out_filter_map_y, 0).transpose().reshape(1, self.input_2_col.shape[0], self.input_2_col.shape[1])
            filter_map = self.input_2_col * filter_reshaped

            # since examples are rolled next to each other we need to unroll then back to ndarray so that we can
            # sum across filters correctly
            total = offset_idx.shape[0] * offset_idx.shape[1]
            sum = numpy.sum(filter_map.reshape(self.input_2_col.shape[0], self.input_2_col.shape[1]), axis=0).reshape(f_z, total).sum(axis=0)
            if self.out_filter_map_x * self.out_filter_map_y == 1:
                sum = numpy.sum(sum)
            out_filter_map.params[f] = sum.reshape(self.depth, self.out_filter_map_x, self.out_filter_map_y)

        self.out_filter_map = out_filter_map
        return out_filter_map

    def im_2_col(self, padded_input):
        # flatten to single matrix
        input_z, input_y, input_x = padded_input.shape
        total_size_for_1_dimension = input_x * input_y
        col_extent = input_x - self.filter_x + 1
        row_extent = input_y - self.filter_y + 1
        padded_input = padded_input.reshape(input_z * input_y, input_x)
        # Parameters
        input_y, input_x = padded_input.shape
        # Get Starting block indices
        start_idx = (numpy.arange(self.filter_y)[:, None] * input_x + numpy.arange(self.filter_x))

        # Get off-setted indices across the height and width of input array
        offset_idx = (numpy.arange(0, row_extent, self.stride)[:, None] * input_x + numpy.arange(0, col_extent, self.stride))
        dimension_offsets = (numpy.arange(input_z).reshape(input_z, 1, 1) * (total_size_for_1_dimension))
        all_indexes = numpy.hstack((start_idx.ravel()[:, None] + offset_idx.ravel()) + dimension_offsets)
        input_matrix = padded_input.take(all_indexes)
        return input_matrix, offset_idx, all_indexes

    def backward(self):
        for f in range(0, len(self.filters)):
            f_z, f_y, f_x = self.filters[f].params.shape
            # gradient and filter will have same shape so just use it in calculations
            out_grad_tiled = numpy.tile(self.out_filter_map.grad[f].reshape(1, f_x * f_y, 1), (f_x * f_y * f_z))
            filter_reshaped = numpy.repeat(self.filters[f].params.reshape(f_z, 1, f_x * f_y),
                                           self.out_filter_map_y * self.out_filter_map_y, 0).transpose().reshape(1, self.input_2_col.shape[0], self.input_2_col.shape[1])

            filter_grad_repeated = numpy.repeat(self.filters[f].grad().reshape(f_z, 1, f_x * f_y),
                                           self.out_filter_map_y * self.out_filter_map_y, 0).transpose().reshape(1, self.input_2_col.shape[0], self.input_2_col.shape[1])

            filter_new_grad = (filter_grad_repeated + (out_grad_tiled * self.input_2_col))

            split_filter_dw = numpy.dsplit(filter_new_grad, f_z)
            for index in range(0, len(split_filter_dw)):
                self.filters[f].grads[index] = split_filter_dw[index].reshape(f_y * f_x, f_y * f_x).sum(axis=1).reshape(f_y, f_x)

            input_dw = (out_grad_tiled * filter_reshaped).reshape(f_y * f_x, f_y * f_x * f_z)
            for index in range(0, self.input_rolled_out_indexes.shape[1]):
                current_grad = self.input_conv.grads.take(self.input_rolled_out_indexes[:, index])
                input_grad = input_dw[:, index]
                numpy.put(self.input_conv.grads, self.input_rolled_out_indexes[:, index], current_grad + input_grad)


    def get_params_and_grads(self):
        return self.input_conv

    def get_filters_and_grads(self):
        return self.filters