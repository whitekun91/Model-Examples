import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_dim, initial_gen_nodes, batch_eps, neg_slope, mnist_size):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.initial_gen_nodes = initial_gen_nodes
        self.batch_eps = batch_eps
        self.neg_slope = neg_slope
        self.mnist_size = mnist_size

        def block(input_dim, output_dim, normalize=True):
            layers = [nn.Linear(input_dim, output_dim)]
            if normalize:
                layers.append(nn.BatchNorm1d(output_dim, self.batch_eps))
            layers.append(nn.LeakyReLU(self.neg_slope, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(self.latent_dim, self.initial_gen_nodes, normalize=False),
            *block(self.initial_gen_nodes, self.initial_gen_nodes * (2 ** 1)),
            *block(self.initial_gen_nodes * (2 ** 1), self.initial_gen_nodes * (2 ** 2)),
            *block(self.initial_gen_nodes * (2 ** 2), self.initial_gen_nodes * (2 ** 3)),
            nn.Linear(self.initial_gen_nodes * (2 ** 3), 1 * self.mnist_size * self.mnist_size),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), 1, self.mnist_size, self.mnist_size)
        return img


class Discriminator(nn.Module):
    def __init__(self, initial_dis_nodes, neg_slope, mnist_size):
        super(Discriminator, self).__init__()
        self.initial_dis_nodes = initial_dis_nodes
        self.neg_slope = neg_slope
        self.mnist_size = mnist_size

        self.model = nn.Sequential(
            nn.Linear(1 * self.mnist_size * self.mnist_size, self.initial_dis_nodes),
            nn.LeakyReLU(self.neg_slope, inplace=True),
            nn.Linear(self.initial_dis_nodes, int(self.initial_dis_nodes / (2 ** 1))),
            nn.LeakyReLU(self.neg_slope, inplace=True),
            nn.Linear(int(self.initial_dis_nodes / (2 ** 1)), 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        flattened = img.view(img.size(0), -1)
        output = self.model(flattened)
        return output
