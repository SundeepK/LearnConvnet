import numpy
from convMatrix import ConvMatrix
import math
import scipy
import numpyUtils
from scipy import signal

class ConvLayer(object):

    def __init__(self, stride, padding, filter_x, filter_y, filter_d, name="", filters=None, bias=None):
        self.stride = stride
        self.padding = padding
        self.filter_d = filter_d
        self.filter_x = filter_x
        self.filter_y = filter_y
        if filters is None:
            self.filters = []
        else:
            self.filters = filters
        bias = numpy.zeros(self.filter_d)
        if bias is None:
            bias.fill(0.1)
        self.bias = ConvMatrix(self.filter_d, 1, 1, bias.copy(), bias.copy())
        self.name = name
        self.input_2_col = None
        self.input_conv_padded = None
        self.im_2_col_indexes = None
        self.out_filter_map_x = None
        self.out_filter_map_y = None
        self.input_conv = None

    @classmethod
    def with_filters(cls, filters, stride, padding):
        obj = cls(stride, padding, filters[0].x, filters[0].y, len(filters), "", filters)
        return obj

    def forward(self, input_matrix):
        self.input_conv_padded = self.get_padded_conv(numpyUtils.pad(input_matrix, self.padding))
        self.input_conv = input_matrix
        self.setup_im_2_col(input_matrix.params, self.input_conv_padded.params)
        self.set_up_filters(input_matrix.params.shape[0]) # pass in matrix depth or z
        self.input_2_col = self.input_conv_padded.params.take(self.im_2_col_indexes)

        # Get all actual indices & index into input array for final output
        out_filter_map = ConvMatrix(self.filter_d, self.out_filter_map_x, self.out_filter_map_y, None, None)
        for f in range(0, self.filter_d):
            f_z, f_y, f_x = self.filters[f].params.shape
            filter_reshaped = numpy.repeat(self.filters[f].params.reshape(f_z, 1, f_x * f_y),
                                           self.out_filter_map_y * self.out_filter_map_y, 0).transpose().reshape(1, self.input_2_col.shape[0], self.input_2_col.shape[1])
            filter_map = self.input_2_col * filter_reshaped

            # since examples are rolled next to each other we need to unroll then back to ndarray so that we can
            # sum across filters correctly
            total = self.out_filter_map_x * self.out_filter_map_y
            sum = numpy.sum(filter_map.reshape(self.input_2_col.shape[0], self.input_2_col.shape[1]), axis=0).reshape(f_z, total).sum(axis=0)
            if self.out_filter_map_x * self.out_filter_map_y == 1:
                sum = numpy.sum(sum)
            out_filter_map.params[f] = sum.reshape(1, self.out_filter_map_x, self.out_filter_map_y) + self.bias.params[f]

        self.out_filter_map = out_filter_map
        return out_filter_map

    def backwards(self, y):
        self.input_conv_padded.grads.fill(0)
        for f in range(0, len(self.filters)):
            f_z, f_y, f_x = self.filters[f].params.shape
            grad = self.out_filter_map.grads[f]
            g_y, g_x = grad.shape

            p_z, p_y, p_x = self.input_conv_padded.params.shape
            grad_tiled = numpy.tile(grad.reshape(1, g_y * g_x), p_z)
            out_grad_tiled = self.input_2_col * grad_tiled
            out_grad_tiled = numpy.hsplit(out_grad_tiled, p_z)

            for index in range(0, f_z):
                self.filters[f].grads[index] = out_grad_tiled[index].sum(axis=1).reshape(f_y, f_x)

            filter_reshaped = numpy.repeat(self.filters[f].params.reshape(f_z, 1, f_x * f_y),
                                           self.out_filter_map_y * self.out_filter_map_y, 0).transpose().reshape(1, self.input_2_col.shape[0], self.input_2_col.shape[1])

            input_dw = (grad_tiled * filter_reshaped).reshape(f_y * f_x, self.im_2_col_indexes.shape[1])
            for index in range(0, self.im_2_col_indexes.shape[1]):
                current_grad = self.input_conv_padded.grads.take(self.im_2_col_indexes[:, index])
                input_grad = input_dw[:, index]
                numpy.put(self.input_conv_padded.grads, self.im_2_col_indexes[:, index], current_grad + input_grad)
            self.bias.grads[f] = grad.sum() + self.bias.grads[f]
        self.input_conv.grads[:] = self.input_conv_padded.grads[:, self.padding:-self.padding, self.padding:-self.padding]

    # only fetch filter indexes as columns if new input array
    # im_2_col_indexes tells us the indexes of the elements the filter window encompasses
    # therefore telling us which elements we need to fetch when multiplying with the filter
    def setup_im_2_col(self, input_matrix, padded_input):
        if self.im_2_col_indexes is None:
            self.init_input_and_indexes(input_matrix, padded_input)

    def init_input_and_indexes(self, input_matrix, padded_input):
        i_z, i_y, i_x = input_matrix.shape
        self.out_filter_map_x = int(math.floor((i_x - self.filter_x + (self.padding * 2)) / self.stride) + 1)
        self.out_filter_map_y = int(math.floor((i_y - self.filter_y + (self.padding * 2)) / self.stride) + 1)
        self.im_2_col_indexes = numpyUtils.im_2_col_indexes(padded_input, self.filter_x,
                                                            self.filter_y, self.stride)

    def get_padded_conv(self, padded_input):
        p_z, p_y, p_x = padded_input.shape
        grad = padded_input.copy()
        grad.fill(0)
        return ConvMatrix(p_z, p_x + self.padding, p_y + self.padding, padded_input, grad)

    def get_input_and_grad(self):
        return self.input_conv

    def get_params_and_grads(self):
        return self.filters

    def get_bias_and_grads(self):
        return [self.bias]

    def out_shape(self):
        return len(self.filters), self.out_filter_map_y, self.out_filter_map_x

    def set_up_filters(self, d):
        if len(self.filters) <= 0:
            for i in range(0, self.filter_d):
                self.filters.append(ConvMatrix(d, self.filter_y, self.filter_x))
