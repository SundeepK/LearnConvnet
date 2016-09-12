import numpy
from matrix import Matrix
import math

class PoolLayer(object):

    def __init__(self, stride, spatial_extent):
        self.stride = stride
        self.spatial_extent = spatial_extent

    def forward(self, inputs):
        input_y, input_x = inputs[0].shape
        col_extent = input_x - self.spatial_extent + 1
        row_extent = input_y - self.spatial_extent + 1

        out_x = math.floor((input_x - self.spatial_extent) / self.stride) + 1
        out_y = math.floor((input_y - self.spatial_extent) / self.stride) + 1

        # Get Starting block indices
        start_idx = (numpy.arange(self.spatial_extent)[:,None]*input_x + numpy.arange(self.spatial_extent))

        # Get off-setted indices across the height and width of input array
        offset_idx = (numpy.arange(0, row_extent, self.stride)[:,None]*input_x + numpy.arange(0, col_extent, self.stride))
        all_indexes = (start_idx.ravel()[:, None] + offset_idx.ravel())
        pooled_out = numpy.empty(len(inputs), dtype=object)
        for f in range(0, len(inputs)):
            filter_map = inputs[f]
            pooled_values = numpy.amax(filter_map.take(all_indexes), axis=0)
            pooled_out[f] = pooled_values.reshape(out_x, out_y)

        return pooled_out
