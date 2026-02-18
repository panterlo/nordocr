"""
Nordic OCR charset and tokenizer.

125 characters covering Swedish, Norwegian, Danish, Finnish, and Icelandic text.
Used by all recognition models (PARSeq, SVTRv2).

Tokenizer layout matches PARSeq convention:
    [EOS] + charset_chars + [BOS, PAD]

The output head predicts (num_tokens - 2) classes: EOS + charset.
BOS and PAD are never predicted by the model.
"""

import torch
from torch.nn.utils.rnn import pad_sequence

# Full Nordic charset: 125 printable characters
NORDIC_CHARSET = (
    # Digits (10)
    "0123456789"
    # ASCII uppercase (26)
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    # ASCII lowercase (26)
    "abcdefghijklmnopqrstuvwxyz"
    # Nordic uppercase (8)
    "ÅÄÖØÆÐÞÜ"
    # Nordic lowercase (8)
    "åäöøæðþü"
    # Punctuation & symbols (47)
    " !\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
    "§°€£«»–—\u2018\u2019\u201c\u201d±×"
)

# Characters that are the focus of diacritical accuracy evaluation
DIACRITICAL_CHARS = set("åäöøæÅÄÖØÆ")

# Confusion pairs: characters commonly confused with each other
CONFUSION_PAIRS = [
    ("a", "å"), ("a", "ä"), ("å", "ä"),
    ("o", "ö"), ("o", "ø"), ("ö", "ø"),
    ("A", "Å"), ("A", "Ä"), ("Å", "Ä"),
    ("O", "Ö"), ("O", "Ø"), ("Ö", "Ø"),
]


class NordicTokenizer:
    """
    Encode/decode text strings to/from integer token sequences.

    Token layout (matches PARSeq convention):
        Index 0:              [EOS]
        Index 1..len(charset): charset characters
        Index -2:             [BOS]
        Index -1:             [PAD]

    The model output head predicts (vocab_size - 2) classes,
    covering EOS + charset only. BOS and PAD are never predicted.
    """

    BOS = "[B]"
    EOS = "[E]"
    PAD = "[P]"

    def __init__(self, charset=NORDIC_CHARSET):
        self.charset = charset
        # Layout: [EOS] + charset + [BOS, PAD]
        specials_first = (self.EOS,)
        specials_last = (self.BOS, self.PAD)
        self._itos = specials_first + tuple(charset) + specials_last
        self._stoi = {s: i for i, s in enumerate(self._itos)}

        self.eos_id = self._stoi[self.EOS]  # 0
        self.bos_id = self._stoi[self.BOS]  # len(charset) + 1
        self.pad_id = self._stoi[self.PAD]  # len(charset) + 2

    @property
    def vocab_size(self):
        """Total vocabulary size including all special tokens."""
        return len(self._itos)

    @property
    def num_output_classes(self):
        """Number of classes the model head predicts (EOS + charset, no BOS/PAD)."""
        return self.vocab_size - 2

    def encode(self, labels, device=None):
        """
        Encode a batch of text labels to padded token tensor.

        Returns: [B, max_len] tensor with BOS prefix, EOS suffix, PAD fill.
        """
        batch = [
            torch.as_tensor(
                [self.bos_id] + [self._stoi.get(c, self.eos_id) for c in y] + [self.eos_id],
                dtype=torch.long,
                device=device,
            )
            for y in labels
        ]
        return pad_sequence(batch, batch_first=True, padding_value=self.pad_id)

    def decode(self, token_dists):
        """
        Decode model output probabilities to text strings.

        Args:
            token_dists: [B, L, C] softmax probabilities over output classes

        Returns:
            (list of decoded strings, list of confidence tensors)
        """
        batch_tokens = []
        batch_probs = []
        for dist in token_dists:
            probs, ids = dist.max(-1)
            ids = ids.tolist()
            try:
                eos_idx = ids.index(self.eos_id)
            except ValueError:
                eos_idx = len(ids)
            ids = ids[:eos_idx]
            probs = probs[:eos_idx + 1]
            tokens = "".join(self._itos[i] for i in ids if i < len(self._itos))
            batch_tokens.append(tokens)
            batch_probs.append(probs)
        return batch_tokens, batch_probs

    def decode_ids(self, ids):
        """Decode a single sequence of token IDs to text."""
        chars = []
        for token_id in ids:
            if token_id == self.eos_id:
                break
            if token_id == self.bos_id or token_id == self.pad_id:
                continue
            if 0 <= token_id < len(self._itos):
                chars.append(self._itos[token_id])
        return "".join(chars)

    def __len__(self):
        return self.vocab_size
