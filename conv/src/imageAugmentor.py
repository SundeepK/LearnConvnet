import cifarUtils
import numpy
from scipy.ndimage import zoom


class ImageAugmentor(object):

    def __init__(self, zoom_factor):
        self.zoom_factor = zoom_factor
        self.img_ndarray = numpy.zeros((3, 32, 32))

    def get(self, img):
        self.setRGBChannels(self.img_ndarray, img)
        original = self.img_ndarray / 255.0-0.5
        flipped = original[...,::-1]
        zoomed = self.zoom(original)
        return [original, flipped, zoomed]

    def setRGBChannels(self, i, x):
        i[0] = x[:, :, 0]
        i[1] = x[:, :, 1]
        i[2] = x[:, :, 2]

    def zoom(self, img):
        if self.zoom_factor == 1:
            return img
        z, h, w = img.shape
        out = zoom(img, (1, self.zoom_factor, self.zoom_factor))
        # crop off any extra pixels at the edges
        trim_top = ((out.shape[1] - h) // 2)
        trim_left = ((out.shape[2] - w) // 2)
        return out[:, trim_top:out.shape[1]-trim_top, trim_left:out.shape[2]-trim_left]
