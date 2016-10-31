import numpy
from convMatrix import ConvMatrix
import math

class ConvNet(object):

    def __init__(self, layers):
        self.layers = layers

    def forward(self, input_conv_matrix):
        activation = self.layers[0].forward(input_conv_matrix)
        for l in range(1, len(self.layers)):
            activation = self.layers[l].forward(activation)
        return activation

    def backward(self, expected_values_y):
        total_layers = len(self.layers)
        loss = self.layers[total_layers - 1].backwards(expected_values_y)
        for l in range(total_layers - 2, 0):
            self.layers[l].backward()
        return loss

    def get_params_and_grads(self):
        params_grads = []
        for l in range(0, len(self.layers)):
            params_grads = params_grads + self.layers[l].get_input_and_grad()

