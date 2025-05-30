import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """Feed-forward MLP block."""

    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.act_fn = F.silu if config.hidden_act == "silu" else F.gelu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.fc2(x)
        return x
