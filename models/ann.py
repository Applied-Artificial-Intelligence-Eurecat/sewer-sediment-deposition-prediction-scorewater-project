import torch.nn as nn


class NeuralNetwork(nn.Module):
    def __init__(self, sequence):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = sequence

    def forward(self, x):
        out = self.linear_relu_stack(x)
        return out
