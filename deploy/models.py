from torch import nn as nn, Tensor


class AE(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()
        self.encoding_layers = nn.Sequential(
            nn.Linear(49, 40),
            nn.ReLU(),
            nn.Linear(40, 30),
            nn.ReLU(),
            nn.Linear(30, 10),
            nn.ReLU()
        )
        self.decoding_layers = nn.Sequential(
            nn.Linear(10, 30),
            nn.ReLU(),
            nn.Linear(30, 40),
            nn.ReLU(),
            nn.Linear(40, 49),
            nn.ReLU()
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


# ANN
class NeuralNetwork(nn.Module):
    def __init__(self, sequence):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = sequence

    def forward(self, x):
        out = self.linear_relu_stack(x)
        return out


arguments_assignment = {'section': 'pvector', 'pipeheight': 'pvector', 'pipewidth': 'pvector',
                        'perimeter': 'pvector', 'Length': 'pvector', 'Velocity': 'pvector',
                        'waterheight': 'pvector', 'flow': 'pvector', 'section_1': 'pvector', 'pipeheight_1': 'pvector',
                        'pipewidth_1': 'pvector', 'perimeter_1': 'pvector',
                        'Length_1': 'pvector', 'Velocity_1': 'pvector', 'waterheight_1': 'pvector', 'flow_1': 'pvector',
                        'section_2': 'pvector',
                        'pipeheight_2': 'pvector', 'pipewidth_2': 'pvector', 'perimeter_2': 'pvector',
                        'Length_2': 'pvector', 'Velocity_2': 'pvector',
                        'waterheight_2': 'pvector', 'flow_2': 'pvector', 'section_3': 'pvector',
                        'pipeheight_3': 'pvector', 'pipewidth_3': 'pvector',
                        'perimeter_3': 'pvector', 'Length_3': 'pvector', 'Velocity_3': 'pvector',
                        'waterheight_3': 'pvector', 'flow_3': 'pvector',
                        'section_4': 'pvector', 'pipeheight_4': 'pvector', 'pipewidth_4': 'pvector',
                        'perimeter_4': 'pvector', 'Length_4': 'pvector',
                        'Velocity_4': 'pvector', 'waterheight_4': 'pvector', 'flow_4': 'pvector',
                        'section_5': 'pvector', 'pipeheight_5': 'pvector',
                        'pipewidth_5': 'pvector', 'perimeter_5': 'pvector', 'Length_5': 'pvector',
                        'Velocity_5': 'pvector', 'waterheight_5': 'pvector',
                        'flow_5': 'pvector', 'neighbourhood': 'pvector', 'amount_rain_mean': 'dvector',
                        'amount_rain_std': 'dvector', 'value_0': 'dvector', 'value_1': 'dvector',
                        'cleaning_applied_0': 'dvector', 'cleaning_applied_1': 'dvector',
                        'amount_rain_mean_1': 'dvector',
                        'amount_rain_std_1': 'dvector', 'value_0_1': 'dvector', 'value_1_1': 'dvector',
                        'cleaning_applied_0_1': 'dvector',
                        'cleaning_applied_1_1': 'dvector', 'amount_rain_mean_2': 'dvector',
                        'amount_rain_std_2': 'dvector',
                        'value_0_2': 'dvector', 'value_1_2': 'dvector', 'cleaning_applied_0_2': 'dvector',
                        'cleaning_applied_1_2': 'dvector', 'amount_rain_mean_3': 'dvector',
                        'amount_rain_std_3': 'dvector',
                        'value_0_3': 'dvector', 'value_1_3': 'dvector', 'cleaning_applied_0_3': 'dvector',
                        'cleaning_applied_1_3': 'dvector', 'amount_rain_mean_4': 'dvector',
                        'amount_rain_std_4': 'dvector',
                        'value_0_4': 'dvector', 'value_1_4': 'dvector', 'cleaning_applied_0_4': 'dvector',
                        'cleaning_applied_1_4': 'dvector', 'amount_rain_mean_5': 'dvector',
                        'amount_rain_std_5': 'dvector',
                        'value_0_5': 'dvector', 'value_1_5': 'dvector', 'cleaning_applied_0_5': 'dvector',
                        'cleaning_applied_1_5': 'dvector'}
