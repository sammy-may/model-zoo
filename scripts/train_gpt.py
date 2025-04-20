import argparse

from model_zoo.autoregressive.helper import AutoRegHelper
from model_zoo.autoregressive.models import GPT
from model_zoo.utils import setup_logger


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train a GPT on Mary Shelley's Frankenstein."
    )

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

    return parser.parse_args()


def main(args):
    logger = setup_logger(args.log_level, args.log_file)
    logger.debug("Training GPT")

    args = {k: v for k, v in vars(args).items() if v is not None}
    gpt_helper = AutoRegHelper(model=GPT, n_epochs=10, **args)
    gpt_helper.load_data()
    gpt_helper.train()


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
