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
            padded_input = padded_input[1:-1, :, :]
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
            padded_input = padded_input[self.stride:-self.stride, :, :]

        # Parameters
        input_y, input_x = input.shape

        # Get Starting block indices
        start_idx = numpy.arange(self.filter_y)[:,None]*input_x + numpy.arange(self.filter_x)

        # Get offsetted indices across the height and width of input array
        offset_idx = numpy.arange(self.out_filter_map_y)[:,None]*input_x + numpy.arange(self.out_filter_map_x)

        # Get all actual indices & index into input array for final output
        out = numpy.take (padded_input,start_idx.ravel()[:,None] + offset_idx.ravel())

