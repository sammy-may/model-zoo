import torch
import torch.nn as nn


class SelfAttentionHead(nn.Module):
    """
    A single head of self-attention.

    Args:
        n_emb (int): Embedding dimension of the input.
        context_len (int): Maximum context length.
        head_size (int): Size of the attention head.
        dropout (float): Dropout rate applied to attention weights.
    """

    def __init__(self, n_emb, context_len, head_size, dropout=0.1):
        super().__init__()

        # k-params
        self.n_emb = n_emb
        self.context_len = context_len
        self.head_size = head_size

        self.key = nn.Linear(n_emb, head_size, bias=False)
        self.query = nn.Linear(n_emb, head_size, bias=False)
        self.value = nn.Linear(n_emb, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(context_len, context_len)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [BATCH, CONTEXT, EMBED]
        key = self.key(x)  # [BATCH, CONTEXT, HEAD_SIZE]
        query = self.query(x)

        attn = query @ key.transpose(-2, -1)  # dot product
        attn *= key.shape[-1] ** (-0.5)  # normalize for stability
        attn = attn.masked_fill(self.tril == 0, float("-inf"))
        attn = torch.nn.functional.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        value = self.value(x)

        out = attn @ value
        return out  # [BATCH, CONTEXT, HEAD_SIZE]


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention mechanism.

    Args:
        num_heads (int): Number of attention heads.
        n_emb (int): Embedding dimension.
        context_len (int): Maximum context length.
        head_size (int): Size of each attention head.
        dropout (float): Dropout rate.
    """

    def __init__(self, num_heads, n_emb, context_len, head_size, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.n_emb = n_emb
        self.context_len = context_len
        self.head_size = head_size
        self.dropout = nn.Dropout(dropout)

        self.heads = nn.ModuleList(
            [
                SelfAttentionHead(n_emb, context_len, head_size, dropout)
                for _ in range(num_heads)
            ]
        )
        self.fc_layer = nn.Linear(head_size * num_heads, n_emb)

    def forward(self, x):
        x = torch.cat([sah(x) for sah in self.heads], dim=-1)
        x = self.fc_layer(x)
        x = self.dropout(x)
        return x


class FeedForward(nn.Module):
    """
    Feed-forward helper network for used inside transformer blocks.

    Args:
        n_emb (int): Embedding dimension.
        dropout (float): Dropout rate.
    """

    def __init__(self, n_emb, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(n_emb, n_emb * 4)
        self.fc2 = nn.Linear(n_emb * 4, n_emb)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """
    Finally, a single transformer block combining multi-head attention and feed-forward layers.

    Args:
        num_heads (int): Number of attention heads.
        n_emb (int): Embedding dimension.
        context_len (int): Maximum context length.
        dropout (float): Dropout rate.
    """

    def __init__(self, num_heads, n_emb, context_len, dropout=0.2):
        super().__init__()
        self.sa = MultiHeadAttention(
            num_heads, n_emb, context_len, n_emb // num_heads, dropout
        )
        self.ffwd = FeedForward(n_emb)

        self.ln1 = nn.LayerNorm(n_emb)
        self.ln2 = nn.LayerNorm(n_emb)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPT(nn.Module):
    """
    Wrap it all together: GPT-style language model with stacked transformer blocks.

    Args:
        vocab_size (int): Size of the vocabulary.
        n_emb (int): Embedding dimension.
        context_len (int): Maximum context length.
        device (str): Device identifier (e.g., 'cuda' or 'cpu').
    """

    def __init__(self, vocab_size, n_emb, context_len, device):
        super(GPT, self).__init__()
        self.vocab_size = vocab_size
        self.n_emb = n_emb
        self.context_len = context_len
        self.device = device

        self.token_embedding = nn.Embedding(vocab_size, n_emb)
        self.position_embedding = nn.Embedding(context_len, n_emb)
        self.ln_f = nn.LayerNorm(n_emb)
        self.lm_head = nn.Linear(n_emb, vocab_size)

        num_heads = 16
        num_layers = 4
        self.transformers = nn.ModuleList(
            [TransformerBlock(num_heads, n_emb, context_len) for _ in range(num_layers)]
        )

    def forward(self, x):
        tok = self.token_embedding(x)
        pos = self.position_embedding(
            torch.arange(self.context_len, device=self.device)
        )
        x = tok + pos
        for t in self.transformers:
            x = t(x)
        x = self.lm_head(self.ln_f(x))
        return x

    def generate(self, x, max_steps):
        for _ in range(max_steps):
            x_crop = x[
                :, -self.context_len :
            ]  # only last N entries are relevant for prediction
            logits = self(x_crop)  #  [1, CONTEXT, VOCAB]
            logits = logits[:, -1, :]  #  take last token only, [1, VOCAB]
            probs = torch.nn.functional.softmax(logits, dim=-1)
            sample = torch.multinomial(probs, num_samples=1)
            x = torch.cat((x, sample), dim=1)
        return x
