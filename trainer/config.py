
from dataclasses import dataclass, field
from typing import List

@dataclass
class TrainingConfig:
    """Configuration settings for model training.

    Attributes:
        batch_size (int): Number of samples per training batch.
        seq_len (int): Length of input sequences.
        epochs (int): Total number of training epochs.
        steps_per_epoch (int): Number of steps (batches) per epoch.
        log_interval (int): Interval for logging loss during training.
        grad_clip_norm (float): Gradient clipping threshold.
        warmup_steps (int): Number of warmup steps for LR scheduler.
        max_lr (float): Peak learning rate for router parameters.
        base_lr (float): Base learning rate for non-router parameters.
        min_lr (float): Final learning rate at the end of decay.
        load_balance_loss_coeff (float): Coefficient for variance term in L_ExpBal.
        alpha1 (float): Coefficient for L_ExpBal loss.
        capacity_factor (float): Capacity factor for token-dropping in MoE.
        checkpoint_dir (str): Directory to save model checkpoints.
        save_interval (int): Interval (in steps) for saving checkpoints.
        eval_fraction (float): Fraction of validation data to evaluate.
        num_local_experts (int): Number of local experts in MoE.
        num_experts_per_tok (int): Number of experts selected per token.
        num_shared_experts (int): Number of shared experts in MoE.
        moe_layers (List[int]): Layers with MoE.
    """
    batch_size: int = 16
    seq_len: int = 128
    epochs: int = 1000
    steps_per_epoch: int = 100
    log_interval: int = 10
    grad_clip_norm: float = 1.0
    warmup_steps: int = 100
    max_lr: float = 0.002  # For router parameters
    base_lr: float = 0.001  # For non-router parameters
    min_lr: float = 0.0001
    load_balance_loss_coeff: float = 5.0
    alpha1: float = 2.0
    capacity_factor: float = 1.2
    checkpoint_dir: str = "checkpoints"
    save_interval: int = 100
    eval_fraction: float = 0.01
    num_local_experts: int = 8
    num_experts_per_tok: int = 2
    num_shared_experts: int = 4
    moe_layers: List[int] = field(default_factory=lambda: [0, 1])