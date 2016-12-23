import numpy
from convMatrix import ConvMatrix
from convoNet import ConvNet
from convNetTrainer import ConvNetTrainer
from convLayer import ConvLayer
from fullyConnectedLayer import FullyConnectedLayer
from poolLayer import PoolLayer
from softMaxLayer import SoftmaxLayer
from reluLayer import ReluLayer
from inputLayer import InputLayer
from scipy.misc import toimage
import cifarUtils
import time
import threading
import sys

MAX_IMAGES_PER_BATCH = 3000


class ConvNNRunner(threading.Thread):

    def __init__(self, training_hook):
        threading.Thread.__init__(self)
        l1 = InputLayer()
        l2 = ConvLayer(1, 2, 5, 5, 16, "l8")
        l3 = ReluLayer()
        l4 = PoolLayer(2, 2, "l4")
        l5 = ConvLayer(1, 2, 5, 5, 20, "l8")
        l6 = ReluLayer()
        l7 = PoolLayer(2, 2, "l7")
        l8 = ConvLayer(1, 2, 5, 5, 20, "l8")
        l9 = ReluLayer()
        l10 = PoolLayer(2, 2, "l10")
        l11 = FullyConnectedLayer()
        l12 = SoftmaxLayer()
        self.layers = [l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12]
        self.convNet = ConvNet(self.layers)
        self.trainer = ConvNetTrainer(0.0001, 0.95, 0.00000001, 4, self.convNet)
        self.training_hook = training_hook
        self.batches = ["./cifar10/data_batch_1",
                        "../cifar10/data_batch_2",
                        "../cifar10/data_batch_3",
                        "../cifar10/data_batch_4",
                        "../cifar10/data_batch_5"]
        self.paused = False
        self.should_stop = False
        self.pause_cond = threading.Condition(threading.Lock())

    def stop(self):
        if self.paused:
            self.resume()
        self.should_stop = True

    def pause(self):
        self.paused = True
        self.pause_cond.acquire()

    def resume(self):
        self.paused = False
        self.pause_cond.notify()
        self.pause_cond.release()

    def run(self):
        for batch in self.batches:
            x, y = cifarUtils.load_CIFAR_batch(batch)
            img_ndarray = numpy.zeros((3, 32, 32))
            for index in range(0, MAX_IMAGES_PER_BATCH):
                # wait if paused
                with self.pause_cond:
                    if self.paused:
                        self.pause_cond.wait()

                if self.should_stop:
                    return

                self.setRGBChannels(img_ndarray, x[index])
                img_ndarray = img_ndarray / 255.0-0.5
                self.train(img_ndarray, y[index])

    def setRGBChannels(self, i, x):
            i[0] = x[:, :, 0]
            i[1] = x[:, :, 1]
            i[2] = x[:, :, 2]

    def train(self, i, y):
        depth, y_input, x_input = i.shape
        stats = self.trainer.train(ConvMatrix(depth, y_input, x_input, i.copy()), y)
        self.training_hook.on_forward_prop(toimage(i), stats)
