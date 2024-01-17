import torch
import torch.nn as nn
import math

class Autoencoder(nn.Module):
    def __init__(self, input_dim, num_layers, bottleneck_size, activation_func=nn.ELU):
        super(Autoencoder, self).__init__()

        assert num_layers % 2 == 0 and num_layers > 2, "Number of layers should be an even number greater than 2."

        def next_power_of_two(x):
            return 2 ** math.ceil(math.log2(x))

        encoder_layers = []
        layer_dimensions = [input_dim]  # Store the input dimension

        # Construct the encoder
        current_dim = input_dim
        for i in range(num_layers // 2 - 1):
            next_dim = next_power_of_two(current_dim // 2)
            encoder_layers.append(nn.Linear(current_dim, next_dim))
            encoder_layers.append(activation_func())
            layer_dimensions.append(next_dim)
            current_dim = next_dim

        # Bottleneck layer
        bottleneck_layer = nn.Linear(current_dim, bottleneck_size)
        encoder_layers.append(bottleneck_layer)
        encoder_layers.append(activation_func())
        layer_dimensions.append(bottleneck_size)

        self.encoder = nn.Sequential(*encoder_layers)

        # Construct the decoder using the reverse of the encoder dimensions
        decoder_layers = []
        layer_dimensions = layer_dimensions[:-1]  # Exclude bottleneck size
        current_dim = bottleneck_size
        for prev_dim in reversed(layer_dimensions):
            decoder_layers.append(nn.Linear(current_dim, prev_dim))
            decoder_layers.append(activation_func())
            current_dim = prev_dim

        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class CustomAE(nn.Module):
    def __init__(self, input_dim):
        super(CustomAE, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.BatchNorm1d(16),
            nn.ELU(),
            nn.Linear(16, 8),
            nn.BatchNorm1d(8),
            nn.ELU(),
            nn.Linear(8, 4),
            nn.BatchNorm1d(4),
            nn.ELU(),
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(4, 8),
            nn.BatchNorm1d(8),
            nn.ELU(),
            nn.Linear(8, 16),
            nn.BatchNorm1d(16),
            nn.ELU(),
            nn.Linear(16, input_dim),
            nn.ELU()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class CustomVAE(nn.Module):
    def __init__(self, input_dim):
        super(CustomVAE, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.BatchNorm1d(16),  # Batch normalization layer
            nn.ELU(),
            nn.Linear(16, 8),
            nn.BatchNorm1d(8),   # Batch normalization layer
            nn.ELU()
        )
        
        self.mean_layer = nn.Linear(8, 4)  # Mean of the latent space
        self.logvar_layer = nn.Linear(8, 4)  # Standard deviation of the latent space

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(4, 8),
            nn.BatchNorm1d(8),   # Batch normalization layer
            nn.ELU(),
            nn.Linear(8, 16),
            nn.BatchNorm1d(16),  # Batch normalization layer
            nn.ELU(),
            nn.Linear(16, input_dim),
            nn.ELU()  # or nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder(x)
        mean, logvar = self.mean_layer(x), self.logvar_layer(x)
        return mean, logvar

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var
