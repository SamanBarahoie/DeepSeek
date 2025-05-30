from dataclasses import dataclass
from typing import List, Optional

@dataclass
class TextConfig:
    """Configuration class for the Transformer model."""
    vocab_size: int
    hidden_size: int
    n_layers: int
    max_seq_len: int
    pad_token_id: int
    num_attention_heads: int
    d_c: int  # Dimension of compressed keys and values per head
    d_c_q: int  # Dimension of compressed queries per head
    d_rh: int  # Dimension of decoupled RoPE queries and shared key per head
    intermediate_size: int
    hidden_act: str = "gelu"  # Activation function for MLP
    rms_norm_eps: float = 1e-5
    num_experts_per_tok: int = 2
    num_shared_experts: int = 4
    num_local_experts: int = 4
    attention_chunk_size: Optional[int] = None
    no_rope_layers: List[int] = None
    moe_layers: List[int] = None
    load_balance_loss_coeff: float = 0.01
    initializer_range: float = 0.02
    lora_rank: int = 4  # LoRA rank for attention
    lora_alpha: float = 8.0  # LoRA scaling factor

    def __post_init__(self):
        """Validates configuration parameters after initialization."""
        if self.no_rope_layers is None:
            self.no_rope_layers = []
        if self.moe_layers is None:
            self.moe_layers = []
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_attention_heads ({self.num_attention_heads})"
            )
        if self.d_c <= 0 or self.d_c_q <= 0 or self.d_rh <= 0:
            raise ValueError(
                f"d_c ({self.d_c}), d_c_q ({self.d_c_q}), and d_rh ({self.d_rh}) must be positive"
            )
        if self.intermediate_size <= 0:
            raise ValueError(f"intermediate_size ({self.intermediate_size}) must be positive")
        if self.hidden_act not in ["silu", "gelu"]:
            raise ValueError(f"hidden_act must be 'silu' or 'gelu', got {self.hidden_act}")
        if self.lora_rank <= 0:
            raise ValueError(f"lora_rank ({self.lora_rank}) must be positive")
        if self.lora_alpha <= 0:
            raise ValueError(f"lora_alpha ({self.lora_alpha}) must be positive")