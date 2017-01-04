import cifarUtils
import numpy
from examples import Examples


class FlippedExamples(Examples):

    def __init__(self, batch):
        super(FlippedExamples, self).__init__(batch)

    def get(self, image_num):
        img, y = Examples.get(self, image_num)
        return img[...,::-1], y
