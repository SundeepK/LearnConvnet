import numpy
from convMatrix import ConvMatrix
import math
from convoNet import ConvNet
from trainResult import TrainingResult
import timeit

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
        forward_time = timeit.default_timer()
        self.convNet.forward(x)
        forward_time = timeit.default_timer() - forward_time
        cost_loss = self.convNet.backwards(y)
        backwards_time = timeit.default_timer() - forward_time
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
                params = pg.params()
                grads = pg.grads()
                l2_decay_loss = numpy.sum(self.l2_decay * ((params * params) / 2))
                l2_grad = self.l2_decay * params
                g = (l2_grad + grads) / self.batch_size
                self.gsum[i] = self.ro * self.gsum[i] + (1-self.ro) * g * g
                dx = - math.sqrt((self.xsum[i] + self.eps)/(self.gsum[i] + self.eps)) * g
                self.xsum[i] = self.ro * self.xsum[i] + (1 - self.ro) * dx * dx
                params[i] += dx
                grads.fill(0)

        return TrainingResult(l2_decay_loss, cost_loss, cost_loss + l2_decay_loss, forward_time, backwards_time)
