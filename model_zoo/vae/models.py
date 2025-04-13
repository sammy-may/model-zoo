import torch
import torch.nn as nn

from model_zoo.utils import setup_logger

logger = setup_logger()


class LinearEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(LinearEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.fc_input = nn.Linear(input_dim, hidden_dim)
        self.fc_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.latent_mean = nn.Linear(hidden_dim, latent_dim)
        self.latent_logvar = nn.Linear(hidden_dim, latent_dim)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.gelu(self.fc_input(x))
        x = self.gelu(self.fc_hidden(x))

        mean = self.latent_mean(x)
        logvar = self.latent_logvar(x)

        return mean, logvar


class LinearDecoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, latent_dim):
        super(LinearDecoder, self).__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.fc_latent = nn.Linear(latent_dim, hidden_dim)
        self.fc_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.gelu(self.fc_latent(x))
        x = self.gelu(self.fc_hidden(x))
        return torch.sigmoid(self.fc_out(x))


class VAE(nn.Module):
    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        mean, logvar = self.encoder(x)
        noise = torch.randn_like(mean)
        z = mean + (torch.exp(0.5 * logvar) * noise)
        reconstruction = self.decoder(z)

        return reconstruction, mean, logvar


class Encoder(nn.Module):
    def __init__(self, in_channels=1, latent_dim=20):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, 16, kernel_size=4, stride=2, padding=1
        )  # 28x28 -> 14x14
        self.conv2 = nn.Conv2d(
            16, 32, kernel_size=4, stride=2, padding=1
        )  # 14x14 -> 7x7
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)  # 7x7 -> 7x7

        self.flatten = nn.Flatten()
        self.fc_mean = nn.Linear(32 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(32 * 7 * 7, latent_dim)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.gelu(self.conv1(x))
        x = self.gelu(self.conv2(x))
        x = self.gelu(self.conv3(x))
        x = self.flatten(x)

        mean = self.fc_mean(x)
        logvar = self.fc_logvar(x)
        return mean, logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim=20, out_channels=1):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 32 * 7 * 7)
        self.gelu = nn.GELU()

        self.deconv1 = nn.ConvTranspose2d(
            32, 32, kernel_size=3, stride=1, padding=1
        )  # 7x7
        self.deconv2 = nn.ConvTranspose2d(
            32, 16, kernel_size=4, stride=2, padding=1
        )  # 7x7 -> 14x14
        self.deconv3 = nn.ConvTranspose2d(
            16, out_channels, kernel_size=4, stride=2, padding=1
        )  # 14x14 -> 28x28

    def forward(self, z):
        x = self.gelu(self.fc(z))
        x = x.view(-1, 32, 7, 7)
        x = self.gelu(self.deconv1(x))
        x = self.gelu(self.deconv2(x))
        x = torch.sigmoid(self.deconv3(x))  # Output pixel values in [0,1]
        return x
