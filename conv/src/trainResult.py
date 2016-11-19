import numpy
from convMatrix import ConvMatrix
import math
import scipy
from scipy import signal

class TrainingResult(object):

    def __init__(self, l2_decay_loss, cost_loss, total_loss, forward_time, backwards_time):
        self.l2_decay_loss = l2_decay_loss
        self.cost_loss = cost_loss
        self.total_loss = total_loss
        self.forward_time = forward_time
        self.backwards_time = backwards_time

    def l2_decay_loss(self):
        return self.l2_decay_loss

    def get_cost_loss(self):
        return self.cost_loss

    def total_loss(self):
        return self.total_loss

    def forward_time(self):
        return self.forward_time

    def backwards_time(self):
        return self.backwards_time
