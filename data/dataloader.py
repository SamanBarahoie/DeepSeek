import logging
from typing import List, Tuple, Optional, Iterator
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset
from transformers import PreTrainedTokenizerFast
from configs.model_config import TextConfig
from trainer.trainer import TrainingConfig

# Configure logging with a consistent format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class TinyStoriesDataset(IterableDataset, Dataset):
    """Dataset for TinyStories, supporting both streaming and non-streaming modes for language modeling."""
    
    def __init__(
        self,
        seq_len: int,
        token_ids: torch.Tensor,
        vocab_size: int,
        is_streaming: bool = True,
        prefetch_size: int = 4096,
    ):
        """
        Initializes the dataset with tokenized data for training or validation.

        Args:
            seq_len: Length of input and target sequences.
            token_ids: Tensor of tokenized data [total_tokens].
            vocab_size: Size of the vocabulary for token validation.
            is_streaming: If True, enables streaming mode with random sampling.
            prefetch_size: Number of indices to pre-sample in streaming mode.

        Raises:
            ValueError: If token_ids length is too short or contains invalid values.
        """
        super().__init__()
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.is_streaming = is_streaming
        self.prefetch_size = prefetch_size

        # Validate and clamp token IDs
        self.token_ids = self._validate_and_clamp_tokens(token_ids)
        self.total_tokens = len(self.token_ids)

        if self.total_tokens < self.seq_len + 1:
            raise ValueError(
                f"Token sequence length ({self.total_tokens}) is too short for seq_len ({self.seq_len} + 1)."
            )

        logging.info(
            f"TinyStoriesDataset initialized: total_tokens={self.total_tokens}, "
            f"seq_len={self.seq_len}, is_streaming={self.is_streaming}, vocab_size={self.vocab_size}"
        )

    def _validate_and_clamp_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Validates token IDs and clamps them to [0, vocab_size - 1].

        Args:
            token_ids: Tensor of token IDs [total_tokens].

        Returns:
            Clamped token IDs tensor.

        Raises:
            ValueError: If token_ids is not a 1D tensor.
        """
        if token_ids.dim() != 1:
            raise ValueError(f"Expected 1D token_ids tensor, got shape {token_ids.shape}")

        invalid_tokens = (token_ids < 0) | (token_ids >= self.vocab_size)
        if invalid_tokens.any():
            invalid_indices = torch.where(invalid_tokens)[0]
            invalid_count = invalid_tokens.sum().item()
            logging.warning(
                f"Found {invalid_count} invalid token IDs at indices {invalid_indices[:10].tolist()}. "
                f"Clamping to [0, {self.vocab_size - 1}]."
            )
            token_ids = torch.clamp(token_ids, 0, self.vocab_size - 1)
        return token_ids

    def __len__(self) -> int:
        """
        Returns the number of samples in non-streaming mode.

        Raises:
            NotImplementedError: If called in streaming mode.
        """
        if self.is_streaming:
            raise NotImplementedError("Length is undefined for streaming datasets.")
        return max(0, self.total_tokens - self.seq_len)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves a sample of input and target sequences at the given index.

        Args:
            idx: Starting index for the sample.

        Returns:
            Tuple of (input_tokens, target_tokens), each of shape [seq_len].

        Raises:
            IndexError: If the index is out of bounds.
            ValueError: If the sample contains invalid token IDs.
        """
        if idx + self.seq_len + 1 > self.total_tokens:
            raise IndexError(
                f"Index {idx} + seq_len {self.seq_len} exceeds total tokens {self.total_tokens}"
            )

        input_tokens = self.token_ids[idx:idx + self.seq_len]
        target_tokens = self.token_ids[idx + 1:idx + self.seq_len + 1]

        self._validate_sample(input_tokens, target_tokens, idx)
        return input_tokens, target_tokens

    def _validate_sample(self, input_tokens: torch.Tensor, target_tokens: torch.Tensor, idx: int) -> None:
        """
        Validates that input and target sequences contain valid token IDs.

        Args:
            input_tokens: Input sequence tensor [seq_len].
            target_tokens: Target sequence tensor [seq_len].
            idx: Starting index of the sample.

        Raises:
            ValueError: If any token IDs are invalid.
        """
        for tokens, name in [(input_tokens, "input"), (target_tokens, "target")]:
            if (tokens < 0).any() or (tokens >= self.vocab_size).any():
                logging.error(
                    f"Invalid {name} tokens at index {idx}: min={tokens.min()}, max={tokens.max()}"
                )
                raise ValueError(f"Invalid {name} tokens at index {idx}: {tokens.tolist()}")

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Iterates over the dataset, yielding samples.

        Yields:
            Tuple of (input_tokens, target_tokens) for each sample.

        Notes:
            In streaming mode, samples are randomly selected with replacement.
            In non-streaming mode, samples are yielded sequentially.
        """
        if not self.is_streaming:
            for idx in range(len(self)):
                yield self[idx]
        else:
            while True:
                indices = torch.randint(
                    0, self.total_tokens - self.seq_len - 1, (self.prefetch_size,), device="cpu"
                )
                for idx in indices:
                    yield self[idx.item()]

def collate_batch(batch: List[Tuple[torch.Tensor, torch.Tensor]], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collates a batch of samples and moves them to the specified device.

    Args:
        batch: List of (input_tokens, target_tokens) tuples.
        device: Target device for the tensors (e.g., 'cuda' or 'cpu').

    Returns:
        Tuple of (input_batch, target_batch), each of shape [batch_size, seq_len].
    """
    input_tokens, target_tokens = zip(*batch)
    input_batch = torch.stack(input_tokens).to(device, non_blocking=True)
    target_batch = torch.stack(target_tokens).to(device, non_blocking=True)
    return input_batch, target_batch

class DataLoaderFactory:
    """Factory for creating DataLoaders for TinyStories training and validation datasets."""
    
    def __init__(
        self,
        model_config: TextConfig,
        training_config: TrainingConfig,
        train_token_file: str = "tokenized-train-samples_vocab-10k.pt",
        valid_token_file: str = "tokenized-valid-samples_vocab-10k.pt",
        tokenizer_file: str = "bpe_tokenizer_fixed",
        pad_token: str = "[PAD]",
        device: str = "cuda",
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        """
        Initializes the factory with configuration for data loading.

        Args:
            model_config: Model configuration with vocab_size and other parameters.
            training_config: Training configuration with batch_size, seq_len, etc.
            train_token_file: Path to tokenized training data.
            valid_token_file: Path to tokenized validation data.
            tokenizer_file: Path to pretrained tokenizer.
            pad_token: Token used for padding.
            device: Device for data loading ('cuda' or 'cpu').
            num_workers: Number of worker processes for data loading.
            pin_memory: If True, enables pinned memory for faster GPU transfer.

        Raises:
            RuntimeError: If tokenizer loading or validation fails.
        """
        self.model_config = model_config
        self.training_config = training_config
        self.train_token_file = train_token_file
        self.valid_token_file = valid_token_file
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.num_workers = max(0, min(num_workers, torch.get_num_threads()))  # Ensure valid num_workers
        self.pin_memory = pin_memory and self.device.type == "cuda"  # Only use pin_memory for CUDA

        # Load and validate tokenizer
        try:
            self.tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_file)
            if self.tokenizer.pad_token is None or self.tokenizer.pad_token != pad_token:
                logging.info(
                    f"Setting pad_token to '{pad_token}' (id={self.tokenizer.convert_tokens_to_ids(pad_token)})"
                )
                self.tokenizer.pad_token = pad_token
                self.tokenizer.save_pretrained(tokenizer_file)

            if self.tokenizer.vocab_size != model_config.vocab_size:
                logging.warning(
                    f"Tokenizer vocab_size ({self.tokenizer.vocab_size}) does not match "
                    f"model vocab_size ({model_config.vocab_size})."
                )
        except Exception as e:
            logging.error(f"Failed to load tokenizer from {tokenizer_file}: {str(e)}")
            raise RuntimeError(f"Tokenizer initialization failed: {str(e)}")

        logging.info(
            f"DataLoaderFactory initialized: device={self.device}, "
            f"vocab_size={self.tokenizer.vocab_size}, num_workers={self.num_workers}"
        )

    def _load_tokens(self, file_path: str) -> torch.Tensor:
        """
        Loads tokenized data from a file.

        Args:
            file_path: Path to the tokenized data file (.pt).

        Returns:
            Tensor of token IDs [total_tokens].

        Raises:
            RuntimeError: If loading fails.
        """
        try:
            tokens = torch.load(file_path, map_location="cpu",weights_only=True)
            if not isinstance(tokens, torch.Tensor):
                raise ValueError(f"Loaded data from {file_path} is not a torch.Tensor")
            return tokens
        except Exception as e:
            logging.error(f"Failed to load tokens from {file_path}: {str(e)}")
            raise RuntimeError(f"Token loading failed: {str(e)}")

    def create_train_loader(self) -> DataLoader:
        """
        Creates a DataLoader for the training dataset.

        Returns:
            DataLoader for training with streaming enabled.
        """
        tokens = self._load_tokens(self.train_token_file)
        dataset = TinyStoriesDataset(
            seq_len=self.training_config.seq_len,
            token_ids=tokens,
            vocab_size=self.model_config.vocab_size,
            is_streaming=True,
            prefetch_size=self.training_config.batch_size * 2,
        )
        return DataLoader(
            dataset,
            batch_size=self.training_config.batch_size,
            pin_memory=self.pin_memory,
            drop_last=True,
            num_workers=self.num_workers,
            collate_fn=lambda batch: collate_batch(batch, self.device),
        )

    def create_valid_loader(self) -> DataLoader:
        """
        Creates a DataLoader for the validation dataset.

        Returns:
            DataLoader for validation with streaming disabled.
        """
        tokens = self._load_tokens(self.valid_token_file)
        dataset = TinyStoriesDataset(
            seq_len=self.training_config.seq_len,
            token_ids=tokens,
            vocab_size=self.model_config.vocab_size,
            is_streaming=False,
        )
        return DataLoader(
            dataset,
            batch_size=self.training_config.batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            collate_fn=lambda batch: collate_batch(batch, self.device),
        )

    def create_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """
        Creates both training and validation DataLoaders.

        Returns:
            Tuple of (train_loader, valid_loader).
        """
        return self.create_train_loader(), self.create_valid_loader()