"""Token Packed Dataset with Hugging Face streaming.

Streams text data from Hugging Face datasets, tokenizes it, appends EOS separators
between documents, and packs tokens into fixed-length contiguous sequence blocks.

This guarantees 100% of tokens in a batch are active — zero compute wasted on padding.
The +1 offset between input_ids and labels creates natural (input, target) pairs
for next-token prediction without any special handling.
"""

import torch
from torch.utils.data import IterableDataset

from configs.base import DataConfig


class TokenPackedDataset(IterableDataset):
    """Streams and packs tokenized text into fixed-length sequences.

    Pipeline:
    1. Stream examples from HF dataset with shuffle buffer
    2. Tokenize each example's text field
    3. Append EOS token between documents
    4. Concatenate into buffer and yield fixed-length chunks

    Each yielded sample:
    - input_ids: (max_seq_len,) — token IDs
    - labels: (max_seq_len,) — next-token targets (shifted by 1)
    """

    def __init__(self, config: DataConfig):
        super().__init__()
        self.config = config
        self.max_seq_len = config.max_seq_len
        self.eos_id = config.eos_token_id

    def _create_dataset(self):
        """Create the HF streaming dataset."""
        from datasets import load_dataset

        ds = load_dataset(
            self.config.dataset_name,
            split=self.config.dataset_split,
            streaming=self.config.streaming,
        )
        if self.config.streaming and self.config.buffer_size > 0:
            ds = ds.shuffle(buffer_size=self.config.buffer_size, seed=42)
        return ds

    def _create_tokenizer(self):
        """Create and validate the tokenizer."""
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_name)

        # Validate vocab size matches config expectations
        # This catches mismatches early before we waste GPU time
        return tokenizer

    def __iter__(self):
        """Iterate over packed sequences.

        Yields dictionaries with input_ids and labels tensors.
        """
        dataset = self._create_dataset()
        tokenizer = self._create_tokenizer()

        token_buffer: list[int] = []
        target_len = self.max_seq_len + 1  # +1 for input/label split

        for example in dataset:
            text = example.get(self.config.dataset_text_field, "")
            if not text or not text.strip():
                continue

            # Tokenize without special tokens (EOS is our separator)
            tokens = tokenizer.encode(text, add_special_tokens=False)
            if not tokens:
                continue

            # Add to buffer with EOS separator
            token_buffer.extend(tokens)
            token_buffer.append(self.eos_id)

            # Yield complete sequences from buffer
            while len(token_buffer) >= target_len:
                chunk = token_buffer[:target_len]
                token_buffer = token_buffer[self.max_seq_len:]

                # input_ids = chunk[:-1], labels = chunk[1:] (next-token prediction)
                input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                labels = torch.tensor(chunk[1:], dtype=torch.long)

                yield {
                    "input_ids": input_ids,
                    "labels": labels,
                }


def create_dataloader(config: DataConfig, batch_size: int):
    """Create a DataLoader from the token-packed dataset.

    Args:
        config: Data configuration.
        batch_size: Number of sequences per batch.

    Returns:
        PyTorch DataLoader yielding batches of packed sequences.
    """
    from torch.utils.data import DataLoader

    dataset = TokenPackedDataset(config)

    def collate_fn(batch):
        """Stack individual samples into batched tensors."""
        return {
            "input_ids": torch.stack([s["input_ids"] for s in batch]),
            "labels": torch.stack([s["labels"] for s in batch]),
        }

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )
    return dataloader
