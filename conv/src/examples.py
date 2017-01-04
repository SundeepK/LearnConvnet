import cifarUtils
import numpy


class Examples(object):

    def __init__(self, batch):
        self.batch = batch
        self.x = None
        self.y = None
        self.img_ndarray = numpy.zeros((3, 32, 32))

    def loadExamples(self):
        self.x, self.y = cifarUtils.load_CIFAR_batch(self.batch)

    def get(self, image_num):
        self.setRGBChannels(self.img_ndarray, self.x[image_num])
        return self.img_ndarray / 255.0-0.5, self.y[image_num]

    def setRGBChannels(self, i, x):
        i[0] = x[:, :, 0]
        i[1] = x[:, :, 1]
        i[2] = x[:, :, 2]

    def deleteExamples(self):
        del self.x
        del self.y
