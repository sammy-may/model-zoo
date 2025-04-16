import argparse

from model_zoo.utils import setup_logger
from model_zoo.vae.vae import VAEHelper


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train a VAE on MNIST.")

    parser.add_argument(
        "--log-level",
        required=False,
        default="INFO",
        type=str,
        help="Level of information printed by the logger",
    )

    parser.add_argument(
        "--log-file", required=False, type=str, help="Name of the log file"
    )

    parser.add_argument(
        "--latent-dim",
        required=False,
        default=32,
        type=int,
        help="Dimensionality of VAE latent space.",
    )

    parser.add_argument(
        "--batch-size",
        required=False,
        default=64,
        type=int,
        help="Training batch size for VAE.",
    )

    return parser.parse_args()


def main(args):
    logger = setup_logger(args.log_level, args.log_file)
    logger.debug("Training VAE")

    vae_helper = VAEHelper(
        latent_dim=args.latent_dim,
        batch_size=args.batch_size,
    )
    vae_helper.load_data()
    vae_helper.train()


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
