import torch.nn as nn
from torch import Tensor


class AutoEncoder(nn.Module):

    def __init__(self, shape, layer1, layer2, layer3, activation):
        super().__init__()
        self.encoding_layers = nn.Sequential(
            nn.Linear(shape, layer1),
            activation,
            nn.Linear(layer1, layer2),
            activation,
            nn.Linear(layer2, layer3),
            activation
        )
        self.decoding_layers = nn.Sequential(
            nn.Linear(layer3, layer2),
            activation,
            nn.Linear(layer2, layer1),
            activation,
            nn.Linear(layer1, shape),
            activation
        )

    def encode(self, features: Tensor) -> Tensor:
        out = self.encoding_layers(features)
        return out

    def decode(self, encoded: Tensor) -> Tensor:
        out = self.decoding_layers(encoded)
        return out

    def forward(self, features: Tensor) -> Tensor:
        encoded = self.encode(features)
        return self.decode(encoded)
