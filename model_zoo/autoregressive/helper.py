import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)

device = (
    torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
)


class CharDataset(Dataset):
    """Custom PyTorch Dataset for character-level language modeling."""

    def __init__(self, chars, block_size):
        """
        Initializes the character dataset.

        Args:
            chars (List[int]): A list of encoded character indices.
            block_size (int): The context length for input sequences.
        """
        self.chars = chars
        self.block_size = block_size

    def __len__(self):
        """A dataset has `N - (block_size + 1)` entries."""
        return len(self.chars) - (self.block_size + 1)

    def __getitem__(self, idx):
        """Instance is a string of length block_size and corresponding target is a string of length block_size shifted right by 1."""
        x = self.chars[idx : idx + self.block_size]
        y = self.chars[idx + 1 : idx + self.block_size + 1]
        return x, y


class AutoRegHelper:
    """
    Helper class for training, evaluating, and inferencing an autoregressive model (e.g. CNN, RNN, Transformer) on a plain text dataset.

    This class loads data, encodes characters, builds dataloaders, and handles training loops
    for models such as RNNs, Transformers, etc.
    """

    def __init__(
        self,
        model: nn.Module,
        n_epochs: int = 10,
        training_files: list[str] = ["data/text/frankenstein.txt"],
        **params,
    ):
        """
        Initializes the AutoRegHelper.

        Args:
            model (nn.Module): Model class (not instance) to be trained.
            n_epochs (int): Number of training epochs.
            training_files (List[str]): List of paths to training text files.
            **params: Additional training parameters like block_size, batch_size.
        """
        self.model = model
        self.n_epochs = n_epochs
        self.training_files = training_files

        self.vocab_size = (
            None  # will infer automatically from the set of chars in all training files
        )
        self.block_size = params.get("block_size", 16)
        self.batch_size = params.get("batch_size", 64)
        self.train_fraction = 0.8
        self.test_fraction = 0.1

        self.device = str(device)
        logger.info(
            "[AutoRegHelper] Training models on device: '{:s}'.".format(self.device)
        )

    def load_data(self):
        """
        Loads and processes the training text data.

        Creates training, test, and validation DataLoaders. Encodes characters and builds vocab mappings.
        """
        text = ""
        for dataset in self.training_files:
            with open(dataset, "r", encoding="utf-8") as f_in:
                new_text = f_in.read()
                logger.debug(
                    "From dataset {:s}, found {:d} raw characters.".format(
                        dataset, len(new_text)
                    )
                )
                text += new_text

        characters = sorted(list(set(text)))
        self.vocab_size = len(characters)

        logger.debug("Found {:d} unique characters".format(self.vocab_size))

        char_to_int = {ch: i for i, ch in enumerate(characters)}
        int_to_char = {i: ch for i, ch in enumerate(characters)}

        self.encode = lambda x: [char_to_int[ch] for ch in x]
        self.decode = lambda x: "".join([int_to_char[i] for i in x])

        data = torch.tensor(self.encode(text), dtype=torch.long).to(device)
        n = int(self.train_fraction * len(data))
        n2 = int((self.train_fraction + self.test_fraction) * len(data))

        self.train_data = DataLoader(
            CharDataset(data[:n], self.block_size),
            batch_size=self.batch_size,
            shuffle=True,
        )
        self.test_data = DataLoader(
            CharDataset(data[n:n2], self.block_size), batch_size=self.batch_size
        )
        self.val_data = DataLoader(
            CharDataset(data[n2:], self.block_size), batch_size=self.batch_size
        )

        logger.debug(
            "Created train/test/val sets with {:d}/{:d}/{:d} instances.".format(
                len(self.train_data), len(self.test_data), len(self.val_data)
            )
        )

    def train(self):
        """
        Trains the autoregressive model using cross-entropy loss.

        Logs training and testing loss per epoch and shows generated sample text from the model.
        """
        model = self.model(
            vocab_size=self.vocab_size,
            n_emb=256,
            context_len=self.block_size,
            device=device,
        )
        model = model.to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()

        # Training Loop: train -> evaluate -> generate
        for epoch in range(self.n_epochs):
            iter = 0
            epoch_loss = 0
            epoch_test_loss = 0

            # Training
            model.train()
            for x, y in tqdm(self.train_data):
                iter += 1
                optimizer.zero_grad()

                logits = model(x)
                BATCH, CONTEXT, VOCAB = logits.shape
                loss = loss_fn(
                    logits.view(BATCH * CONTEXT, VOCAB),
                    y.view(BATCH * CONTEXT),  # reshape to place nice with BCE
                )
                epoch_loss += loss.item()

                loss.backward()
                optimizer.step()

            # Evaluation
            model.eval()
            with torch.no_grad():
                for x, y in tqdm(self.test_data):
                    logits = model(x)
                    BATCH, CONTEXT, VOCAB = logits.shape
                    loss = loss_fn(
                        logits.view(BATCH * CONTEXT, VOCAB), y.view(BATCH * CONTEXT)
                    )
                    epoch_test_loss += loss.item()

            # Normalize losses
            epoch_loss *= 1.0 / (len(self.train_data))
            epoch_test_loss *= 1.0 / (len(self.test_data))

            logger.info(
                "Epoch {:d}: train/test loss: {:.3f}/{:.3f}.".format(
                    epoch, epoch_loss, epoch_test_loss
                )
            )

            # Generation! It's alive! (Frankenstein reference)
            a = model.generate(
                torch.tensor(self.encode("Today is a")).reshape(1, -1).to(device), 200
            )
            logger.info(
                "Message from model : \n \t {:s}.".format(
                    self.decode(torch.squeeze(a).cpu().numpy())
                )
            )
