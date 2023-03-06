# Sewer sediment deposition prediction

The repository encapsulates the code that generates a two-phase machine learning model, combining an autoencoder to reduce the dimensionality 
of physical (and static) features of sewer pipes, and an artificial neural network that combines the reduced features with dynamic to classify 
future sediment occupation ranges.
The development has been conducted within the project SCOREwater. This project has received funding from the European Union’s Horizon 2020 Research and Innovation Programme under grant agreement no 820751. A paper explaining the study behind the code is on the way.


# Dependencies, Installation, and Usage

- Requires Python 3.7.
- Dependencies listed inside the requirements.txt file.
- Architecture configuration inside the conf.yaml file.
- In order to use the code, data is mandatory. The needed format is specified in models/input example.txt.
- An example on how to use these models in a deploy stage can be seen in /deploy directory.

To use the already saved models, they can be loaded into a python script using the following code:
```
from models import AE, NeuralNetwork
import torch
import torch.nn as nn

def load_models():
    ae = AE()
    sequence = nn.Sequential(nn.Linear(46, 10), nn.Softsign(), nn.Linear(10, 10), nn.Softsign(), nn.Linear(10, 1),
                             nn.Sigmoid())
    ann5 = NeuralNetwork(sequence)
    ann10 = NeuralNetwork(sequence)
    ann15 = NeuralNetwork(sequence)
    ann20 = NeuralNetwork(sequence)
    ae.load_state_dict(torch.load('trained_models/AE'))
    ann5.load_state_dict(torch.load('trained_models/model_threshold_5'))
    ann10.load_state_dict(torch.load('trained_models/model_threshold_10'))
    ann15.load_state_dict(torch.load('trained_models/model_threshold_15'))
    ann20.load_state_dict(torch.load('trained_models/model_threshold_20'))
    return [ae, ann5, ann10, ann15, ann20]
```


# Citing

Paper on the way.


Copyright 2023 Eurecat