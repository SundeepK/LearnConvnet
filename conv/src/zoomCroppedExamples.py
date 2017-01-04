import cifarUtils
import numpy
from examples import Examples
from scipy.ndimage import zoom


class ZoomCroppedExamples(Examples):

    def __init__(self, batch, zoom_factor):
        super(ZoomCroppedExamples, self).__init__(batch)
        self.zoom_factor = zoom_factor

    def get(self, image_num):
        img, y = Examples.get(self, image_num)
        if self.zoom_factor == 1:
            return img
        z, h, w = img.shape
        out = zoom(img, (1, self.zoom_factor, self.zoom_factor))
        # crop off any extra pixels at the edges
        trim_top = ((out.shape[1] - h) // 2)
        trim_left = ((out.shape[2] - w) // 2)
        return out[:, trim_top:out.shape[1]-trim_top, trim_left:out.shape[2]-trim_left], y

