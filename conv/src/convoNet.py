import numpy
from convMatrix import ConvMatrix
import math
import json


class ConvNet(object):

    def __init__(self, layers):
        self.layers = layers

    def forward(self, input_conv_matrix):
        activation = self.layers[0].forward(input_conv_matrix)
        if len(self.layers) > 0:
            for l in range(1, len(self.layers), 1):
                activation = self.layers[l].forward(activation)
        return activation

    def backwards(self, expected_y):
        total_layers = len(self.layers)
        loss = self.layers[total_layers - 1].backwards(expected_y)
        if len(self.layers) > 0:
            for l in range(total_layers - 2, -1, -1):
                    self.layers[l].backwards(expected_y)
        return loss

    def get_params_and_grads(self):
        params_grads = []
        for l in range(0, len(self.layers)):
            params_grads = params_grads + self.layers[l].get_params_and_grads()
            params_grads = params_grads + self.layers[l].get_bias_and_grads()
        return params_grads

    def get_json(self):
        json_array = []
        for layer in self.layers:
            json_array.append(layer.to_dict())
        return json.dumps({'CNN': json_array})
