import torch
import torch.nn as nn

from model_zoo.utils import setup_logger

logger = setup_logger()


class VAE(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean, logvar = self.encoder(x)
        noise = torch.randn_like(logvar)
        z = mean + noise * (torch.exp(0.5 * logvar))
        reconstruction = self.decoder(z)

        return reconstruction, mean, logvar

    def generate(self, N: int = 1) -> torch.Tensor:
        z = torch.randn(N, self.encoder.fc_mean.out_features)
        return self.decoder(z)


class Encoder(nn.Module):
    def __init__(self, in_channels: int = 1, latent_dim: int = 32):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, 16, kernel_size=4, stride=2, padding=1
        )  # 28x28 -> 14x14
        self.batch_norm1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(
            16, 32, kernel_size=4, stride=2, padding=1
        )  # 14x14 -> 7x7
        self.batch_norm2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)  # 7x7 -> 7x7
        self.batch_norm3 = nn.BatchNorm2d(32)

        self.flatten = nn.Flatten()
        self.fc_mean = nn.Linear(32 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(32 * 7 * 7, latent_dim)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.batch_norm1(self.conv1(x)))
        x = self.activation(self.batch_norm2(self.conv2(x)))
        x = self.activation(self.batch_norm3(self.conv3(x)))
        x = self.flatten(x)

        mean = self.fc_mean(x)
        logvar = self.fc_logvar(x)

        return mean, logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim: int = 32, out_channels: int = 1):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 32 * 7 * 7)
        self.activation = nn.GELU()

        self.deconv1 = nn.ConvTranspose2d(
            32, 32, kernel_size=3, stride=1, padding=1
        )  # 7x7
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.deconv2 = nn.ConvTranspose2d(
            32, 16, kernel_size=4, stride=2, padding=1
        )  # 7x7 -> 14x14
        self.batch_norm2 = nn.BatchNorm2d(16)
        self.deconv3 = nn.ConvTranspose2d(
            16, out_channels, kernel_size=4, stride=2, padding=1
        )  # 14x14 -> 28x28

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.fc(x))

        x = x.view(-1, 32, 7, 7)

        x = self.activation(self.batch_norm1(self.deconv1(x)))
        x = self.activation(self.batch_norm2(self.deconv2(x)))
        return torch.sigmoid(self.deconv3(x))  # bound pixel values in [0,1]
