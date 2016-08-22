import numpy
from matrix import Matrix
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
                self.filters.append(Matrix(3, filter_x, filter_y))
        else:
            self.depth = d
            self.filter_x = filter_x
            self.filter_y = filter_y
            self.filters = filters
        self.out_filter_map_x = math.floor(((self.input_x - self.filter_x + (self.padding * 2))) / self.stride) + 1
        self.out_filter_map_y = math.floor(((self.input_y - self.filter_y + (self.padding * 2))) / self.stride) + 1

    @classmethod
    def with_filters(cls, filters, input_x, input_y, stride, padding):
        obj = cls(input_x, input_y, stride, padding, filters[0].x, filters[0].y, len(filters), filters)
        return obj

    def forward(self, input):
        padded_input = numpy.pad(input, pad_width=self.padding, mode='constant', constant_values=0)
        if padded_input.shape[2] != input.shape[2]:
            padded_input = padded_input[self.padding:-self.padding, :, :]
        out_filter_map = Matrix.with_matrix(numpy.zeros((self.depth, self.out_filter_map_x, self.out_filter_map_y)))
        padded_x = padded_input.shape[1]
        padded_y = padded_input.shape[2]
        for f in range(0, len(self.filters)):
            map_y = 0
            for y in range(0, padded_y, self.stride):
                if y + self.filter_y > padded_y:
                    break
                map_x = 0
                for x in range(0, padded_x, self.stride):
                    current_stride_sum = 0
                    if x + self.filter_x > padded_x:
                        break
                    for d in range(0, self.filters[f].m.shape[2]):
                        current_stride_sum += numpy.sum(padded_input[d, y:(y + self.filter_y), x:(x + self.filter_x)] * self.filters[f].m[d, :, :])
                    out_filter_map.m[f, map_y, map_x] = current_stride_sum
                    map_x += 1
                map_y += 1
        return out_filter_map

    def vectorized_forward(self, input):
        padded_input = numpy.pad(input, pad_width=self.padding, mode='constant', constant_values=0)
        if padded_input.shape[2] != input.shape[2]:
            padded_input = padded_input[1:-1, :, :]

        # flatten to single matrix
        input_z, input_y, input_x = padded_input.shape
        input_total_size = input_x * input_y

        col_extent = input_x - self.filter_x + 1
        row_extent = input_y - self.filter_y + 1

        padded_input = padded_input.reshape(input_z * input_y, input_x)

        # Parameters
        input_y, input_x = padded_input.shape

        # Get Starting block indices
        start_idx = (numpy.arange(self.filter_y)[:,None]*input_x + numpy.arange(self.filter_x))

        # Get off-setted indices across the height and width of input array
        offset_idx = (numpy.arange(0, row_extent, self.stride)[:,None]*input_x + numpy.arange(0, col_extent, self.stride))

        dimension_offsets = (numpy.arange(input_z).reshape(input_z, 1, 1) * (input_total_size))
        all_indexes = numpy.empty((input_z, input_z * self.filter_y, input_z * self.filter_x), dtype=numpy.int)
        all_indexes[:] = (start_idx.ravel()[:, None] + offset_idx.ravel()) + dimension_offsets
        all_indexes = numpy.hstack(all_indexes)

        input_matrix = padded_input.take(all_indexes)

        # Get all actual indices & index into input array for final output
        out_filter_map = numpy.empty(len(self.filters), dtype=object)
        for f in range(0, len(self.filters)):
            f_y, f_x, f_z = self.filters[f].m.shape
            filter_map = input_matrix * numpy.repeat(self.filters[f].m.reshape(f_z, 1, f_z * f_y), 9, 0).transpose().reshape(1, input_matrix.shape[0], input_matrix.shape[1])
            sum = numpy.sum(filter_map.reshape(input_matrix.shape[0], input_matrix.shape[1]).transpose().ravel().reshape(f_z, f_z * f_y, f_z * f_x), axis=2).sum(axis=0)
            out_filter_map[f] = sum.reshape(self.depth, self.out_filter_map_x, self.out_filter_map_y)

        return out_filter_map
