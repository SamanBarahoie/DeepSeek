import torch
from torch.utils.data import Dataset, DataLoader

class FinetunerLoader:
    """
    A single class to load tokenized .pt files into PyTorch DataLoaders for fine-tuning.

    Example:
        loader = FinetunerLoader(
            train_path="train_tokenized.pt",
            eval_path="eval_tokenized.pt",
            batch_size=16,
            shuffle_train=True,
            num_workers=2
        )
        train_loader, eval_loader = loader.load()
    """
    class _CustomDataset(Dataset):
        def __init__(self, tensor_dict):
            self.input_ids = tensor_dict["input_ids"]
            self.attention_mask = tensor_dict["attention_mask"]
            self.labels = tensor_dict["labels"]
            self.length = self.input_ids.size(0)

        def __len__(self):
            return self.length

        def __getitem__(self, idx):
            return {
                "input_ids": self.input_ids[idx],
                "attention_mask": self.attention_mask[idx],
                "labels": self.labels[idx]
            }

    def __init__(
        self,
        train_path: str,
        eval_path: str,
        batch_size: int = 16,
        shuffle_train: bool = True,
        num_workers: int = 2
    ):
        self.train_path = train_path
        self.eval_path = eval_path
        self.batch_size = batch_size
        self.shuffle_train = shuffle_train
        self.num_workers = num_workers

    def load(self):
        """
        Loads .pt files and returns train and eval DataLoaders.

        Returns:
            tuple: (train_loader, eval_loader)
        """
        # Load the tensor dicts
        train_tensors = torch.load(self.train_path,weights_only=True)
        eval_tensors = torch.load(self.eval_path,weights_only=True)

        # Wrap into datasets
        train_dataset = self._CustomDataset(train_tensors)
        eval_dataset = self._CustomDataset(eval_tensors)

        # Create DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle_train,
            num_workers=self.num_workers
        )
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

        return train_loader, eval_loader
