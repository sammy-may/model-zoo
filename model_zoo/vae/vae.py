import torch
import torch.nn as nn
import torchvision

import matplotlib.pyplot as plt

from model_zoo.utils import setup_logger
from model_zoo.vae.models import VAE, Encoder, Decoder

logger = setup_logger()


class VAEHelper:
    def __init__(
        self,
        latent_dim: int,
        batch_size: int,
        n_epochs: int = 100,
        data_path: str = "data/",
    ):
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.data_path = data_path

    def load_data(self):
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
        reco_term = nn.functional.binary_cross_entropy(x_reco, x, reduction="mean")
        kl_term = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
        return reco_term + beta * kl_term

    @staticmethod
    def print_image(x: torch.Tensor, filepath: str):
        """Save image to a pdf file specified by `filepath`."""
        fig, ax = plt.subplots()
        plt.imshow(x, cmap="gray")
        plt.axis("off")
        plt.savefig(filepath, format="pdf", bbox_inches="tight", pad_inches=0)
        plt.close()

    def train(self):
        model = VAE(Encoder(), Decoder())
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

        for epoch in range(self.n_epochs):
            loss_epoch = 0
            model.train()
            for X, y in self.train_loader:
                optimizer.zero_grad()
                X_reco, mean, logvar = model(X)

                loss = self.elbo_loss(
                    X, X_reco, mean, logvar, 0.1 if epoch == 0 else 0.3
                )
                loss_epoch += loss.item()

                loss.backward()
                optimizer.step()

            model.eval()
            loss_epoch *= 1.0 / (len(self.train_loader))
            print(epoch, loss_epoch)

            with torch.no_grad():
                self.print_image(
                    X[-1, 0], self.data_path + "/vae_output/orig_{:d}.pdf".format(epoch)
                )
                self.print_image(
                    X_reco[-1, 0],
                    self.data_path + "/vae_output/reco_{:d}.pdf".format(epoch),
                )
                for i in range(3):
                    self.print_image(
                        model.generate()[0, 0],
                        self.data_path
                        + "/vae_output/gen_{:d}_{:d}.pdf".format(epoch, i),
                    )


if __name__ == "__main__":
    vae = VAEHelper(latent_dim=32, batch_size=64)
    vae.load_data()
    vae.train()
