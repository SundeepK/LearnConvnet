import numpy
from convmatrix import ConvMatrix
import math
import scipy
from scipy import signal

class ConvLayer(object):

    def __init__(self, inputX, inputY, stride, padding, filter_x, filter_y, d, filters=None):
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
                self.filters.append(ConvMatrix(3, filter_x, filter_y))
        else:
            self.depth = d
            self.filter_x = filter_x
            self.filter_y = filter_y
            self.filters = filters
        self.out_filter_map_x = math.floor(((self.input_x - self.filter_x + (self.padding * 2))) / self.stride) + 1
        self.out_filter_map_y = math.floor(((self.input_y - self.filter_y + (self.padding * 2))) / self.stride) + 1

    @classmethod
    def with_filters(cls, filters, input_x, input_y, stride, padding):
        obj = cls(input_x, input_y, stride, padding, filters[0].y, filters[0].x, len(filters), filters)
        return obj

    def forward(self, input):
        padded_input = numpy.pad(input, pad_width=self.padding, mode='constant', constant_values=0)
        if padded_input.shape[2] != input.shape[2]:
            padded_input = padded_input[1:-1, :, :]

        p_z, p_y, p_x = padded_input.shape
        self.input = ConvMatrix(p_z, p_x, p_y, padded_input)
        input_matrix, offset_idx, all_indexes = self.input_data(padded_input)

        # Get all actual indices & index into input array for final output
        out_filter_map = numpy.empty(len(self.filters), dtype=object)
        for f in range(0, len(self.filters)):
            f_z, f_y, f_x = self.filters[f].params.shape
            filter_reshaped = numpy.repeat(self.filters[f].params.reshape(f_z, 1, f_x * f_y),
                                           self.out_filter_map_y * self.out_filter_map_y, 0).transpose().reshape(1, input_matrix.shape[0], input_matrix.shape[1])
            filter_map = input_matrix * filter_reshaped

            # since examples are rolled next to each other we need to unroll then back to ndarray so that we can
            # sum across filters correctly
            total = offset_idx.shape[0] * offset_idx.shape[1]
            sum = numpy.sum(filter_map.reshape(input_matrix.shape[0], input_matrix.shape[1]), axis=0).reshape(f_z, total).sum(axis=0)
            if self.out_filter_map_x * self.out_filter_map_y == 1:
                sum = numpy.sum(sum)
            out = sum.reshape(self.depth, self.out_filter_map_x, self.out_filter_map_y)
            o_z, o_y, o_x = out.shape
            out_filter_map[f] = ConvMatrix(o_z, o_x, o_y, out)

        self.out_filter_map = out_filter_map
        return out_filter_map

    def input_data(self, padded_input):
        # flatten to single matrix
        input_z, input_y, input_x = padded_input.shape
        input_total_size = input_x * input_y
        col_extent = input_x - self.filter_x + 1
        row_extent = input_y - self.filter_y + 1
        padded_input = padded_input.reshape(input_z * input_y, input_x)
        # Parameters
        input_y, input_x = padded_input.shape
        # Get Starting block indices
        start_idx = (numpy.arange(self.filter_y)[:, None] * input_x + numpy.arange(self.filter_x))
        # Get off-setted indices across the height and width of input array
        offset_idx = (numpy.arange(0, row_extent, self.stride)[:, None] * input_x + numpy.arange(0, col_extent, self.stride))
        dimension_offsets = (numpy.arange(input_z).reshape(input_z, 1, 1) * (input_total_size))
        all_indexes = numpy.hstack((start_idx.ravel()[:, None] + offset_idx.ravel()) + dimension_offsets)
        input_matrix = padded_input.take(all_indexes)
        return input_matrix, offset_idx, all_indexes

    @property
    def backward(self):
        input_matrix, offset_idx, all_indexes = self.input_data(self.input.params)

        # Get all actual indices & index into input array for final output
        out_filter_map = numpy.empty(len(self.filters), dtype=object)
        for f in range(0, len(self.filters)):
            f_z, f_y, f_x = self.filters[f].params.shape
            # gradient and filter will have same shape so just use it in calculations
            out_grad_tiled = numpy.tile(self.out_filter_map[f].grad.reshape(1, f_x * f_y, 1), (f_x * f_y * f_z))
            filter_reshaped = numpy.repeat(self.filters[f].params.reshape(f_z, 1, f_x * f_y),
                                           self.out_filter_map_y * self.out_filter_map_y, 0).transpose().reshape(1, input_matrix.shape[0], input_matrix.shape[1])

            filter_reshaped_dw = numpy.repeat(self.filters[f].grad().reshape(f_z, 1, f_x * f_y),
                                           self.out_filter_map_y * self.out_filter_map_y, 0).transpose().reshape(1, input_matrix.shape[0], input_matrix.shape[1])

            filter_dw = (filter_reshaped_dw + (out_grad_tiled * input_matrix))

            split_filter_dw = numpy.dsplit(filter_dw, f_z)
            for index in range(0, len(split_filter_dw)):
                self.filters[f].grads[index] = split_filter_dw[index].reshape(f_y * f_x, f_y * f_x).sum(axis=1).reshape(f_y, f_x)

            # input_dw = (out_grad_tiled * filter_reshaped).reshape(f_y * f_x, f_y * f_x * f_z)
            # input_dw_split = numpy.hsplit(input_dw, f_z)
            #
            # for index in range(0, len(input_dw_split)):
            #     input_dw_split[index] = input_dw_split[index].transpose()
            #
            #
            # input_dw = numpy.concatenate(input_dw_split, axis=0)
            input_dw = (out_grad_tiled * filter_reshaped).reshape(f_y * f_x, f_y * f_x * f_z)
            # input_dw = concatenate.reshape(f_z, f_y * f_x, f_y * f_x)
            print(input_dw[:, 0])
            # print(out_grad_tiled * filter_reshaped)
            for index in range(0, all_indexes.shape[1]):
                if 1 <= index < all_indexes.shape[1]:
                    grads = numpy.arange(f_x - 1, f_x * f_y, f_x)
                    in_dws = numpy.arange(0, f_x * f_y, f_x)

                    take_grad_index = all_indexes[:, index - 1].take(grads)
                    grad_take = self.input.grads.take(take_grad_index)
                    input_dw_take = input_dw[:, index].take(in_dws)

                    accumilated = grad_take + input_dw_take
                    print("take " + str(grads) + "grads indexes:" + str(take_grad_index) + " values at indexes: " + str(grad_take))
                    print("take " + str(in_dws) + "in_dws indexes:" + str(in_dws) + " values at indexes: " + str(input_dw_take))
                    print("accumilated: " + str(accumilated))
                    print("\n")
                    numpy.put(self.input.grads, all_indexes[:, index], input_dw[:, index])
                    numpy.put(self.input.grads, take_grad_index, accumilated)
                    print("self.input.grads: \n" + str(self.input.grads))
                    print("\n")
                elif index == 0:
                    numpy.put(self.input.grads, all_indexes[:, index], input_dw[:, index])
                    print("self.input.grads: \n" + str(self.input.grads))


        self.out_filter_map = out_filter_map
        return out_filter_map

    def get_params_and_grads(self):
        return self.input

    def get_filters_and_grads(self):
        return self.filters