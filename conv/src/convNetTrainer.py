import numpy
from convMatrix import ConvMatrix
import math
from convoNet import ConvNet

class ConvNetTrainer(object):

    def __init__(self, l2_decay, ro, eps, batch_size, convNet):
        self.l2_decay = l2_decay
        self.ro = ro
        self.eps = eps
        self.k = 0
        self.gsum = []
        self.xsum = []
        self.convNet = convNet
        self.batch_size = batch_size

    def train(self, x, y):
        l2_decay_loss = 0
        self.convNet.forward(x)
        self.convNet.backward(y)
        self.k += 1

        if self.k % self.batch_size == 0:
            params_and_grads = self.convNet.get_params_and_grads()
            if len(self.gsum) == 0:
                for i in range(0, len(params_and_grads)):
                    pg = params_and_grads[i]
                    self.xsum.append(ConvMatrix(pg.d, pg.y, pg.x, numpy.zeros((pg.d, pg.y, pg.x))))
                    self.gsum.append(ConvMatrix(pg.d, pg.y, pg.x, numpy.zeros((pg.d, pg.y, pg.x))))

            for i in range(0, len(params_and_grads)):
                pg = params_and_grads[i]
                p = pg.params()
                g = pg.grads()
                l2_decay_loss = numpy.sum(self.l2_decay * ((p * p) / 2))
                l2_grad = self.l2_decay * p
                gij = (l2_grad + g) / self.batch_size