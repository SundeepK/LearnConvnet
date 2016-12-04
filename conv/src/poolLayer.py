import numpy
from convMatrix import ConvMatrix
import math

class PoolLayer(object):

    def __init__(self, stride, spatial_extent):
        self.stride = stride
        self.spatial_extent = spatial_extent

    def forward(self, input_matrix):
        self.input = input_matrix
        input_z, input_y, input_x = input_matrix.params.shape
        col_extent = input_x - self.spatial_extent + 1
        row_extent = input_y - self.spatial_extent + 1

        out_x = math.floor((input_x - self.spatial_extent) / self.stride) + 1
        out_y = math.floor((input_y - self.spatial_extent) / self.stride) + 1

        # Get Starting block indices
        start_idx = (numpy.arange(self.spatial_extent)[:,None]*input_x + numpy.arange(self.spatial_extent))

        # Get off-setted indices across the height and width of input array
        offset_idx = (numpy.arange(0, row_extent, self.stride)[:,None]*input_x + numpy.arange(0, col_extent, self.stride))
        all_indexes = (start_idx.ravel()[:, None] + offset_idx.ravel())
        pooled_out = numpy.empty((int(input_z), int(out_y), int(out_x)))
        self.pooled_out_max_indices = numpy.empty((int(input_z), int(out_y), int(out_x)), dtype=int)
        for f in range(0, input_z):
            filter_map = input_matrix.params[f]
            selected_vales = filter_map.take(all_indexes)
            pooled_values = numpy.amax(selected_vales, axis=0)
            pooled_out[f] = pooled_values.reshape(int(out_x), int(out_y))
            # store indexes of max values from input
            self.pooled_out_max_indices[f] = numpy.diagonal(numpy.take(all_indexes, numpy.argmax(selected_vales, axis=0), axis=0)).reshape(int(out_x), int(out_y))

        self.out = ConvMatrix(input_z, out_y, out_x, pooled_out)
        return self.out

    def backwards(self, y):
        self.input.grads.fill(0)
        out_z, out_y, out_x = self.pooled_out_max_indices.shape
        for f in range(0, out_z):
            chained = numpy.take(self.input.grads[f], self.pooled_out_max_indices[f]).reshape(int(out_y), int(out_x)) + self.out.grads[f]
            numpy.put(self.input.grads[f], self.pooled_out_max_indices[f], chained)

    def get_bias_and_grads(self):
        return []


    def get_params_and_grads(self):
        return []
