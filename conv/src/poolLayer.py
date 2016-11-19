import numpy
from convMatrix import ConvMatrix
import math

class PoolLayer(object):

    def __init__(self, stride, spatial_extent):
        self.stride = stride
        self.spatial_extent = spatial_extent

    def forward(self, input):
        input_z, input_y, input_x = input.params.shape
        col_extent = input_x - self.spatial_extent + 1
        row_extent = input_y - self.spatial_extent + 1

        out_x = math.floor((input_x - self.spatial_extent) / self.stride) + 1
        out_y = math.floor((input_y - self.spatial_extent) / self.stride) + 1

        # Get Starting block indices
        start_idx = (numpy.arange(self.spatial_extent)[:,None]*input_x + numpy.arange(self.spatial_extent))

        # Get off-setted indices across the height and width of input array
        offset_idx = (numpy.arange(0, row_extent, self.stride)[:,None]*input_x + numpy.arange(0, col_extent, self.stride))
        all_indexes = (start_idx.ravel()[:, None] + offset_idx.ravel())
        pooled_out = numpy.empty((input_z, out_y, out_x), dtype=object)
        for f in range(0, input_z):
            filter_map = input.params[f]
            pooled_values = numpy.amax(filter_map.take(all_indexes), axis=0)
            pooled_out[f] = pooled_values.reshape(out_x, out_y)

        return ConvMatrix(input_z, out_y, out_x, pooled_out)

    def backwards(self, y):
        pass


    def get_params_and_grads(self):
        return []
