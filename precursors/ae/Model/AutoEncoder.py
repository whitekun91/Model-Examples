"""
MNIST fully-connected autoencoder.

Data flow (normalized inputs in [-1, 1], Tanh on decoder output):
    x [B,1,H,W] -> flatten [B, H*W] -> Encoder -> z [B, latent_dim]
    -> Decoder -> [B, H*W] -> reshape -> x_hat [B,1,H,W]
"""

import torch.nn as nn


class Encoder(nn.Module):
    """Maps flattened image to latent vector (no activation on the bottleneck)."""

    def __init__(self, input_dim, hidden_dim, latent_dim, neg_slope):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.neg_slope = neg_slope

        # Linear -> LeakyReLU -> Linear(latent)
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.LeakyReLU(self.neg_slope, inplace=True),
            nn.Linear(self.hidden_dim, self.latent_dim),
        )

    def forward(self, x):
        flat = x.view(x.size(0), -1)
        return self.model(flat)


class Decoder(nn.Module):
    """Maps latent vector back to a flattened image, then reshapes to [B,1,H,W]."""

    def __init__(self, latent_dim, hidden_dim, output_dim, neg_slope, mnist_size):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.neg_slope = neg_slope
        self.mnist_size = mnist_size

        # Linear -> LeakyReLU -> Linear -> Tanh (match normalized pixel range)
        self.model = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.LeakyReLU(self.neg_slope, inplace=True),
            nn.Linear(self.hidden_dim, self.output_dim),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.model(z)
        return out.view(out.size(0), 1, self.mnist_size, self.mnist_size)


class AutoEncoder(nn.Module):
    """
    Encoder + decoder for reconstruction loss (e.g. MSE between x and x_hat).

    Shapes:
        forward(x): x [B,1,H,H] -> x_hat [B,1,H,H]
        encode(x):  x -> z [B, latent_dim]
        decode(z):  z -> x_hat [B,1,H,H]
    """

    def __init__(self, mnist_size, hidden_dim, latent_dim, neg_slope):
        super(AutoEncoder, self).__init__()
        self.mnist_size = mnist_size
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.neg_slope = neg_slope
        self.input_dim = 1 * mnist_size * mnist_size

        self.encoder = Encoder(
            self.input_dim, self.hidden_dim, self.latent_dim, self.neg_slope
        )
        self.decoder = Decoder(
            self.latent_dim,
            self.hidden_dim,
            self.input_dim,
            self.neg_slope,
            self.mnist_size,
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)
