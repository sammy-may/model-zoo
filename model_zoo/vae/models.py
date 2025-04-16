import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class VAE(nn.Module):
    """
    Generic Variational Autoencoder (VAE) combining an encoder and decoder.

    Encodes input data into a latent distribution, samples from it using the reparameterization
    trick, then reconstructs the data via the decoder.
    """

    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        """
        Initialize the VAE with an encoder and decoder module.

        Args:
            encoder (nn.Module): Module that encodes input into latent mean and log-variance.
            decoder (nn.Module): Module that decodes latent vectors back to image space.
        """
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass through the VAE.

        Encodes the input to a latent space, samples from the latent distribution,
        and reconstructs the input from the sampled latent vector.

        Args:
            x (torch.Tensor): Input tensor, typically a batch of images.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - Reconstructed images
                - Latent mean
                - Latent log-variance
        """
        mean, logvar = self.encoder(x)
        noise = torch.randn_like(logvar)
        z = mean + noise * (torch.exp(0.5 * logvar))
        reconstruction = self.decoder(z)

        return reconstruction, mean, logvar

    def generate(self, N: int = 1) -> torch.Tensor:
        """
        Generate new samples from the prior distribution.

        Args:
            N (int, optional): Number of samples to generate. Defaults to 1.

        Returns:
            torch.Tensor: Batch of generated images.
        """
        z = torch.randn(N, self.encoder.fc_mean.out_features)
        return self.decoder(z)


class Encoder(nn.Module):
    """
    Convolutional encoder network for a Variational Autoencoder.

    Compresses an input image into a latent distribution (mean and log-variance).
    """

    def __init__(self, in_channels: int = 1, latent_dim: int = 32):
        """
        Initialize the encoder architecture.

        Args:
            in_channels (int, optional): Number of input channels. Defaults to 1 for grayscale.
            latent_dim (int, optional): Size of the latent space. Defaults to 32.
        """
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
        """
        Encode input into latent mean and log-variance.

        Args:
            x (torch.Tensor): Batch of input images.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - Latent mean tensor
                - Latent log-variance tensor
        """
        x = self.activation(self.batch_norm1(self.conv1(x)))
        x = self.activation(self.batch_norm2(self.conv2(x)))
        x = self.activation(self.batch_norm3(self.conv3(x)))
        x = self.flatten(x)

        mean = self.fc_mean(x)
        logvar = self.fc_logvar(x)

        return mean, logvar


class Decoder(nn.Module):
    """
    Convolutional decoder network for a Variational Autoencoder.

    Maps latent vectors back to image space.
    """

    def __init__(self, latent_dim: int = 32, out_channels: int = 1):
        """
        Initialize the decoder architecture.

        Args:
            latent_dim (int, optional): Size of the latent space. Defaults to 32.
            out_channels (int, optional): Number of output image channels. Defaults to 1 for grayscale.
        """
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
        """
        Decode latent vectors into reconstructed images.

        Args:
            x (torch.Tensor): Latent vector(s) sampled from the latent distribution.

        Returns:
            torch.Tensor: Reconstructed image(s) with pixel values in [0, 1].
        """
        x = self.activation(self.fc(x))

        x = x.view(-1, 32, 7, 7)

        x = self.activation(self.batch_norm1(self.deconv1(x)))
        x = self.activation(self.batch_norm2(self.deconv2(x)))
        return torch.sigmoid(self.deconv3(x))  # bound pixel values in [0,1]
