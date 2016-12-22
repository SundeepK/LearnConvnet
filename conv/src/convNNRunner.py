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

MAX_IMAGES_PER_BATCH = 3000

class ConvNNRunner(object):

    def __init__(self, training_hook):
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
        self.current_index = 0
        self.current_batch_number = 0
        self.batches = ["./cifar10/data_batch_1", 
                        "../cifar10/data_batch_2",
                        "../cifar10/data_batch_3",
                        "../cifar10/data_batch_4",
                        "../cifar10/data_batch_5"]
        self.paused = True

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False
        self.start()

    def start(self):
        for index in range(self.current_batch_number, len(self.batches)):
            x, y = cifarUtils.load_CIFAR_batch(self.batches[index])
            img_ndarray = numpy.zeros((3, 32, 32))
            for index in range(self.current_index, MAX_IMAGES_PER_BATCH):
                if self.paused:
                    return
                self.setRGBChannels(img_ndarray, x[index])
                img_ndarray = img_ndarray / 255.0-0.5
                self.train(img_ndarray, y[index])
                self.current_index += 1
            self.current_batch_number += 1
            self.current_index %= MAX_IMAGES_PER_BATCH


    def setRGBChannels(self, i, x):
            i[0] = x[:, :, 0]
            i[1] = x[:, :, 1]
            i[2] = x[:, :, 2]

    def train(self, i, y):
        depth, y_input, x_input = i.shape
        stats = self.trainer.train(ConvMatrix(depth, y_input, x_input, i.copy()), y)
        self.training_hook.onForwardProp(toimage(i), stats)
