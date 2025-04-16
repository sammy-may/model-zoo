import logging

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision

from model_zoo.vae.models import VAE, Decoder, Encoder

logger = logging.getLogger(__name__)


class VAEHelper:
    """
    Helper class for training and evaluating a Variational AutoEncoder (VAE) on the MNIST dataset.

    TODO:
        - make dataset configurable
        - provide options for using Vector Quantization and Finite Scalar Quantization
    """

    def __init__(
        self,
        latent_dim: int,
        batch_size: int,
        n_epochs: int = 100,
        data_path: str = "data/",
    ):
        """
        Initialize the VAEHelper instance with configuration parameters.

        Args:
            latent_dim (int): Dimensionality of the latent space.
            batch_size (int): Batch size for training and evaluation.
            n_epochs (int, optional): Number of training epochs. Defaults to 100.
            data_path (str, optional): Path to store or load MNIST data and outputs. Defaults to "data/".
        """
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.data_path = data_path

    def load_data(self):
        """
        Load the MNIST dataset and initialize training and test data loaders.

        Downloads the dataset to the specified path if not already present.
        TODO:
            - include options for data augmentation
        """
        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
            ]
        )

        train_data = torchvision.datasets.MNIST(
            self.data_path, transform=transform, train=True, download=True
        )
        test_data = torchvision.datasets.MNIST(
            self.data_path, transform=transform, train=False, download=True
        )

        self.train_loader = torch.utils.data.DataLoader(
            dataset=train_data, batch_size=self.batch_size, shuffle=True, num_workers=4
        )
        self.test_loader = torch.utils.data.DataLoader(
            dataset=test_data, batch_size=self.batch_size, shuffle=False, num_workers=4
        )

    @staticmethod
    def elbo_loss(
        x: torch.Tensor,
        x_reco: torch.Tensor,
        mean: torch.Tensor,
        logvar: torch.Tensor,
        beta: float,
    ) -> torch.Tensor:
        """
        Compute the Evidence Lower Bound (ELBO) loss.

        Args:
            x (torch.Tensor): Original input data.
            x_reco (torch.Tensor): Reconstructed output from the decoder.
            mean (torch.Tensor): Mean of the latent distribution.
            logvar (torch.Tensor): Log variance of the latent distribution.
            beta (float): Weight for the KL divergence term.

        Returns:
            torch.Tensor: The total ELBO loss.
        """
        reco_term = nn.functional.binary_cross_entropy(x_reco, x, reduction="mean")
        kl_term = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
        return reco_term + beta * kl_term

    @staticmethod
    def print_image(x: torch.Tensor, filepath: str):
        """
        Save a tensor image as a grayscale PDF file.

        Args:
            x (torch.Tensor): Image tensor to be saved.
            filepath (str): File path where the image will be saved.
        """
        fig, ax = plt.subplots()
        plt.imshow(x, cmap="gray")
        plt.axis("off")
        plt.savefig(filepath, format="pdf", bbox_inches="tight", pad_inches=0)
        plt.close()

    def train(self):
        """
        Train the VAE model using the loaded MNIST dataset.

        The method performs training over a fixed number of epochs, logs the loss per epoch,
        and saves images of original, reconstructed, and generated digits to PDF files at each epoch.
        """
        model = VAE(Encoder(), Decoder())
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

        # Train for fixed number of epochs
        for epoch in range(self.n_epochs):
            loss_epoch = 0
            loss_epoch_test = 0
            beta = min(0.03 * (epoch + 1), 0.3)

            # Iterate through training set
            model.train()
            for X, _ in self.train_loader:
                optimizer.zero_grad()
                X_reco, mean, logvar = model(X)

                loss = self.elbo_loss(X, X_reco, mean, logvar, beta)
                loss_epoch += loss.item()

                loss.backward()
                optimizer.step()

            # Iterate through testing set
            model.eval()
            with torch.no_grad():
                for X, _ in self.test_loader:
                    X_reco, mean, logvar = model(X)

                    loss = self.elbo_loss(X, X_reco, mean, logvar, beta)
                    loss_epoch_test += loss.item()

            loss_epoch *= 1.0 / (len(self.train_loader))
            loss_epoch_test *= 1.0 / (len(self.test_loader))
            logger.info(
                "Epoch {:d}: train/test loss: {:.4f}/{:.4f}".format(
                    epoch, loss_epoch, loss_epoch_test
                )
            )

            # Save images of original, reconstructed, and K generated from sampling in the latent space.
            K = 3
            with torch.no_grad():
                self.print_image(
                    X[-1, 0], self.data_path + "/vae_output/orig_{:d}.pdf".format(epoch)
                )
                self.print_image(
                    X_reco[-1, 0],
                    self.data_path + "/vae_output/reco_{:d}.pdf".format(epoch),
                )
                for i in range(K):
                    self.print_image(
                        model.generate()[0, 0],
                        self.data_path
                        + "/vae_output/gen_{:d}_{:d}.pdf".format(epoch, i),
                    )


if __name__ == "__main__":
    vae = VAEHelper(latent_dim=32, batch_size=64)
    vae.load_data()
    vae.train()
