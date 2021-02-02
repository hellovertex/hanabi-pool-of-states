import torch
from torch.nn import Linear, ReLU, Softmax
import numpy as np
import random


# loop: sample architecture(*), eval_score(Jacobian), add to list
# (*): type of layers, number of layers, size of layers
# two to

def get_model(observation_size,
              num_actions,
              num_hidden_layers,
              layer_size
):
    # layer_size could be replaced with layer_sizes[i]
    in_features = [observation_size] + [layer_size for _ in range(num_hidden_layers)]
    out_features = [layer_size for _ in range(num_hidden_layers)] + [num_actions]

    layers = []
    for i in range(num_hidden_layers + 1):
        layers.append(Linear(in_features[i], out_features[i]))
        layers.append(ReLU())

    return torch.nn.Sequential(*layers)

def random_model(observation_size=1024, num_actions=20, max_hidden_layers=5):
    # create between 1 and 5 hidden layers, each of them having half the number of weights as the previous
    num_hidden_layers = random.randint(1, max_hidden_layers)
    hidden_layer_sizes = [random.choice([128, 256, 512, 1024])]

    for i in range(1, num_hidden_layers):
        hidden_layer_sizes.append(int(hidden_layer_sizes[i - 1] / 2))

    # input and output sizes for each fc-layer
    in_sizes = [observation_size]
    out_sizes = []
    activations = []

    for i in range(num_hidden_layers):
        out_sizes += [hidden_layer_sizes[i]]
        in_sizes += [hidden_layer_sizes[i]]
        activations += [ReLU]
    out_sizes += [num_actions]
    activations += [Softmax]

    layers = []

    for i in range(num_hidden_layers + 1):
        layers.append(Linear(in_sizes[i], out_sizes[i]))
        layers.append(activations[i]())

    return torch.nn.Sequential(*layers)


def gen_model(observation_size=1024, num_actions=20,
              max_hidden_layers=None, sizes_first_layer=None):
    # create between 1 and 5 hidden layers, each of them having half the number of weights as the previous
    if not max_hidden_layers: max_hidden_layers = [i for i in range(1, 5)]
    if not sizes_first_layer: sizes_first_layer = [128, 256, 512, 1024]

    for num_hidden_layers in max_hidden_layers:
        for size_first_layer in sizes_first_layer:
            hidden_layer_sizes = [size_first_layer]
            [hidden_layer_sizes.append(int(hidden_layer_sizes[i - 1] / 2)) for i in range(1, num_hidden_layers)]

            # input and output sizes for each fc-layer
            in_sizes = [observation_size]
            out_sizes = []
            activations = []

            for i in range(num_hidden_layers):
                out_sizes += [hidden_layer_sizes[i]]
                in_sizes += [hidden_layer_sizes[i]]
                activations += [ReLU]
            out_sizes += [num_actions]
            # activations += [Softmax]
            activations += [ReLU]

            layers = []

            for i in range(num_hidden_layers + 1):
                layers.append(Linear(in_sizes[i], out_sizes[i]))
                layers.append(activations[i]())

            yield torch.nn.Sequential(*layers)

