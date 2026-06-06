"""Chat-formatted SFT Dataset for QLoRA post-training.

Downloads and merges multiple HuggingFace instruction/chat datasets,
normalizes to a unified messages format, applies Gemma 4's chat template,
tokenizes, and builds labels with -100 masking (only assistant tokens
contribute to loss).

Supports:
- Multi-dataset mixing with per-dataset sample limits
- Thinking mode toggle via system prompt injection
- Gemma 4 chat template via tokenizer.apply_chat_template
- Train/eval split with configurable ratio
- Robust loss masking via prefix-comparison method
"""

import random
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader

from configs.base import DataConfig


class ChatSFTDataset(Dataset):
    """Chat-formatted SFT dataset with loss masking.

    Each sample is a multi-turn conversation formatted with Gemma's chat template.
    Labels use -100 for all non-assistant tokens (user turns, system prompts, special tokens).
    Only the model's response tokens contribute to the cross-entropy loss.

    Loss masking uses the prefix-comparison method:
    1. Tokenize the full conversation (all turns)
    2. Tokenize the prompt-only version (all turns except last assistant response)
    3. Tokens beyond the prompt length belong to the assistant → unmask those
    This is robust to any chat template format and doesn't depend on string matching.
    """

    def __init__(
        self,
        samples: list[dict],
        tokenizer,
        max_seq_len: int = 4096,
        enable_thinking: bool = False,
        thinking_prompt: str = "",
        fast_prompt: str = "",
    ):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.enable_thinking = enable_thinking
        self.thinking_prompt = thinking_prompt
        self.fast_prompt = fast_prompt

        self._tokenized = self._pre_tokenize()

    def _format_messages(self, messages: list[dict]) -> list[dict]:
        """Prepare messages with optional system prompt injection."""
        has_system = any(m["role"] == "system" for m in messages)
        formatted = []
        if not has_system:
            sys_prompt = self.thinking_prompt if self.enable_thinking else self.fast_prompt
            if sys_prompt:
                formatted.append({"role": "system", "content": sys_prompt})
        formatted.extend(messages)
        return formatted

    def _pre_tokenize(self) -> list[dict]:
        """Pre-tokenize all samples with loss masking via prefix comparison.

        For each conversation:
        1. full_ids = tokenize(all turns including assistant response)
        2. prompt_ids = tokenize(all turns EXCEPT the last assistant response)
        3. prompt_len = len(prompt_ids)
        4. labels = [-100]*prompt_len + full_ids[prompt_len:]
        This robustly identifies assistant tokens regardless of template format.
        """
        tokenized = []
        n_skipped_no_asst = 0
        n_skipped_too_short = 0

        for sample in self.samples:
            messages = sample.get("messages", [])
            if not messages:
                continue

            formatted = self._format_messages(messages)

            # Check tokenizer supports chat template
            if not hasattr(self.tokenizer, "apply_chat_template"):
                # Fallback: simple tokenization, mask everything after last user turn
                item = self._fallback_tokenize(formatted)
                if item is not None:
                    tokenized.append(item)
                continue

            # Find the last assistant message index
            last_asst_idx = -1
            for i in range(len(formatted) - 1, -1, -1):
                if formatted[i]["role"] == "assistant":
                    last_asst_idx = i
                    break

            if last_asst_idx == -1:
                n_skipped_no_asst += 1
                continue

            # Build prompt-only messages (everything up to last assistant response)
            prompt_messages = [
                m for i, m in enumerate(formatted) if i < last_asst_idx
            ]
            # Also include assistant messages that aren't the last one
            # (multi-turn: earlier assistant turns are part of the prompt context)
            # Actually, we want ALL turns except the LAST assistant response
            # This way, earlier assistant turns are masked (they're "given") but
            # the model still sees them as context.

            # Full conversation (with generation prompt=False so it ends after last turn)
            try:
                full_text = self.tokenizer.apply_chat_template(
                    formatted,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            except Exception:
                continue

            # Prompt = full conversation minus the last assistant response
            # Build messages without the last assistant response
            prompt_only = []
            for i, m in enumerate(formatted):
                if i == last_asst_idx:
                    break  # Skip the last assistant response
                prompt_only.append(m)

            try:
                prompt_text = self.tokenizer.apply_chat_template(
                    prompt_only,
                    tokenize=False,
                    add_generation_prompt=True,  # Add the start marker for model turn
                )
            except Exception:
                continue

            # Tokenize both
            full_enc = self.tokenizer(
                full_text,
                max_length=self.max_seq_len,
                truncation=True,
                padding=False,
            )
            prompt_enc = self.tokenizer(
                prompt_text,
                max_length=self.max_seq_len,
                truncation=True,
                padding=False,
            )

            full_ids = full_enc["input_ids"]
            prompt_ids = prompt_enc["input_ids"]

            # Prompt length = number of tokens in the prompt
            # Everything after that is assistant response
            prompt_len = len(prompt_ids)

            # If prompt is longer than full (shouldn't happen but guard),
            # or full is too short, skip
            if prompt_len >= len(full_ids):
                n_skipped_too_short += 1
                continue

            # Build labels: -100 for prompt, real ids for assistant response
            labels = [-100] * prompt_len + full_ids[prompt_len:]

            # Verify at least some tokens are unmasked
            n_unmasked = sum(1 for l in labels if l != -100)
            if n_unmasked == 0:
                n_skipped_too_short += 1
                continue

            attention_mask = [1] * len(full_ids)

            tokenized.append({
                "input_ids": full_ids,
                "labels": labels,
                "attention_mask": attention_mask,
            })

        if n_skipped_no_asst > 0:
            print(f"    Skipped {n_skipped_no_asst} samples (no assistant response)")
        if n_skipped_too_short > 0:
            print(f"    Skipped {n_skipped_too_short} samples (prompt too long or empty response)")

        return tokenized

    def _fallback_tokenize(self, messages: list[dict]) -> Optional[dict]:
        """Fallback tokenization when chat template is unavailable.

        Simple approach: tokenize all text, mask everything except
        the last assistant message content.
        """
        parts = []
        last_asst_content = None
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            parts.append(f"{role}: {content}")
            if role == "assistant":
                last_asst_content = content

        if last_asst_content is None:
            return None

        full_text = "\n".join(parts)
        enc = self.tokenizer(
            full_text,
            max_length=self.max_seq_len,
            truncation=True,
            padding=False,
        )
        input_ids = enc["input_ids"]

        # Rough masking: find the assistant response by encoding it separately
        asst_enc = self.tokenizer(
            f"assistant: {last_asst_content}",
            max_length=self.max_seq_len,
            truncation=True,
            padding=False,
        )
        asst_len = len(asst_enc["input_ids"])

        # Unmask the last asst_len tokens
        labels = [-100] * len(input_ids)
        start = max(0, len(input_ids) - asst_len)
        for i in range(start, len(input_ids)):
            labels[i] = input_ids[i]

        n_unmasked = sum(1 for l in labels if l != -100)
        if n_unmasked == 0:
            return None

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": [1] * len(input_ids),
        }

    def __len__(self) -> int:
        return len(self._tokenized)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self._tokenized[idx]
        return {
            "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
            "labels": torch.tensor(item["labels"], dtype=torch.long),
            "attention_mask": torch.tensor(item["attention_mask"], dtype=torch.long),
        }


def _load_and_normalize_dataset(
    dataset_name: str,
    split: str,
    n_samples: int,
    category: str,
) -> list[dict]:
    """Load a HF dataset and normalize to messages format.

    Handles various common formats:
    - {"messages": [...]} — already in messages format
    - {"conversations": [...]} — sharegpt format
    - {"prompt": ..., "completion": ...} — instruction/completion format
    - {"instruction": ..., "output": ...} — alpaca format
    """
    from datasets import load_dataset

    print(f"  Loading {dataset_name} (split={split}, n={n_samples}, category={category})")
    ds = load_dataset(dataset_name, split=split, trust_remote_code=True)

    samples = []
    for example in ds:
        # Format 1: Already in messages format
        if "messages" in example and isinstance(example["messages"], list):
            messages = example["messages"]
        # Format 2: ShareGPT conversations format
        elif "conversations" in example and isinstance(example["conversations"], list):
            messages = _convert_sharegpt(example["conversations"])
        # Format 3: Prompt/completion format
        elif "prompt" in example and "completion" in example:
            messages = [
                {"role": "user", "content": example["prompt"]},
                {"role": "assistant", "content": example["completion"]},
            ]
        # Format 4: Alpaca instruction format
        elif "instruction" in example:
            instruction = example["instruction"]
            inp = example.get("input", "")
            output = example.get("output", "")
            if inp:
                content = f"{instruction}\n\n{inp}"
            else:
                content = instruction
            messages = [
                {"role": "user", "content": content},
                {"role": "assistant", "content": output},
            ]
        else:
            continue

        # Filter: need at least one user and one assistant message
        roles = [m["role"] for m in messages]
        if "user" in roles and "assistant" in roles:
            samples.append({"messages": messages})

        if n_samples > 0 and len(samples) >= n_samples:
            break

    print(f"    → {len(samples)} samples loaded")
    return samples


def _convert_sharegpt(conversations: list[dict]) -> list[dict]:
    """Convert ShareGPT format to messages format."""
    role_map = {
        "human": "user", "user": "user",
        "gpt": "assistant", "assistant": "assistant",
        "system": "system", "tool": "tool",
    }
    messages = []
    for turn in conversations:
        from_role = turn.get("from", turn.get("role", ""))
        value = turn.get("value", turn.get("content", ""))
        role = role_map.get(from_role, from_role)
        messages.append({"role": role, "content": value})
    return messages


def _pad_collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
    """Collate function that pads sequences to the same length within a batch."""
    max_len = max(item["input_ids"].size(0) for item in batch)

    input_ids_list = []
    labels_list = []
    attention_mask_list = []

    for item in batch:
        seq_len = item["input_ids"].size(0)
        pad_len = max_len - seq_len

        input_ids_list.append(torch.cat([
            item["input_ids"],
            torch.zeros(pad_len, dtype=torch.long),
        ]))
        labels_list.append(torch.cat([
            item["labels"],
            torch.full((pad_len,), -100, dtype=torch.long),
        ]))
        attention_mask_list.append(torch.cat([
            item["attention_mask"],
            torch.zeros(pad_len, dtype=torch.long),
        ]))

    return {
        "input_ids": torch.stack(input_ids_list),
        "labels": torch.stack(labels_list),
        "attention_mask": torch.stack(attention_mask_list),
    }


def create_sft_dataloaders(
    config: DataConfig,
    tokenizer,
) -> tuple[DataLoader, DataLoader]:
    """Create train and eval DataLoaders from the configured dataset mixture."""
    print("[Data] Loading dataset mixture...")
    all_samples = []

    for dataset_name, split, n_samples, category in config.datasets:
        samples = _load_and_normalize_dataset(dataset_name, split, n_samples, category)
        all_samples.extend(samples)

    print(f"[Data] Total samples loaded: {len(all_samples)}")

    random.seed(42)
    random.shuffle(all_samples)

    n_eval = max(1, int(len(all_samples) * config.test_split_ratio))
    n_train = len(all_samples) - n_eval

    train_samples = all_samples[:n_train]
    eval_samples = all_samples[n_train:]

    print(f"[Data] Split: {n_train} train, {n_eval} eval")

    train_ds = ChatSFTDataset(
        samples=train_samples,
        tokenizer=tokenizer,
        max_seq_len=config.max_seq_len,
        enable_thinking=config.enable_thinking,
        thinking_prompt=config.thinking_system_prompt,
        fast_prompt=config.fast_system_prompt,
    )
    eval_ds = ChatSFTDataset(
        samples=eval_samples,
        tokenizer=tokenizer,
        max_seq_len=config.max_seq_len,
        enable_thinking=config.enable_thinking,
        thinking_prompt=config.thinking_system_prompt,
        fast_prompt=config.fast_system_prompt,
    )

    print(f"[Data] Tokenized: {len(train_ds)} train, {len(eval_ds)} eval")

    train_loader = DataLoader(
        train_ds,
        batch_size=1,
        shuffle=True,
        collate_fn=_pad_collate_fn,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=1,
        shuffle=False,
        collate_fn=_pad_collate_fn,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    return train_loader, eval_loader
