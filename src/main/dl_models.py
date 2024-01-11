import torch
import torch.nn as nn
import math

class Autoencoder(nn.Module):
    def __init__(self, input_dim, num_layers, bottleneck_size, activation_func=nn.ELU):
        super(Autoencoder, self).__init__()

        assert num_layers % 2 == 0 and num_layers > 2, "Number of layers should be an even number greater than 2."

        def next_power_of_two(x):
            return 2 ** math.ceil(math.log2(x))

        layers = []
        current_dim = input_dim

        # Construct the encoder
        for i in range(num_layers // 2 - 1):
            next_dim = next_power_of_two(current_dim // 2)  # Calculate the next power of two
            layers.append(nn.Linear(current_dim, next_dim))
            layers.append(activation_func())
            current_dim = next_dim

        # Bottleneck layer
        layers.append(nn.Linear(current_dim, bottleneck_size))
        layers.append(activation_func())

        self.encoder = nn.Sequential(*layers)

        layers = []

        # Construct the decoder
        current_dim = bottleneck_size
        for i in range(num_layers // 2 - 1):
            next_dim = next_power_of_two(current_dim * 2)  # Calculate the next power of two
            layers.append(nn.Linear(current_dim, min(next_dim, input_dim)))  # Ensure dimension does not exceed input_dim
            layers.append(activation_func())
            current_dim = next_dim

        layers.append(nn.Linear(current_dim, input_dim))
        layers.append(activation_func())

        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
