# models/layers/rotary.py
import torch
import torch.nn as nn
from typing import Optional
from configs.model_config import TextConfig

def apply_rope_matrix(x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """
    Apply Rotary Position Embedding (RoPE) to tensor `x` using angles `theta`.

    Args:
        x: Tensor of shape [batch, seq_len, dim] or [batch, seq_len, heads, dim]
        theta: Tensor of shape [batch, seq_len, dim//2] or [seq_len, dim//2]

    Returns:
        Tensor of same shape as `x` - after RoPE rotation.
    """
    if x.dim() == 4:
        batch, seq_len, heads, dim = x.shape
    elif x.dim() == 3:
        batch, seq_len, dim = x.shape
        heads = None
    else:
        raise ValueError(f"Expected 3D or 4D input, got {x.shape}")

    assert dim % 2 == 0, "Embedding dimension must be even"
    half_dim = dim // 2

    # Ensure theta matches batch dimension
    if theta.dim() == 2:
        theta = theta.unsqueeze(0).expand(batch, seq_len, half_dim)

    # Reshape x into pairs
    if heads is not None:
        x = x.view(batch * seq_len * heads, dim)  # Flatten for rotation
    x_pairs = x.view(-1, half_dim, 2)  # [batch*seq_len*heads, half_dim, 2]

    # Split into even and odd
    x_even, x_odd = x_pairs.unbind(dim=-1)

    # Get cos and sin
    cos_theta = theta.cos()  # [batch, seq_len, half_dim]
    sin_theta = theta.sin()
    if heads is not None:
        cos_theta = cos_theta.unsqueeze(2).expand(batch, seq_len, heads, half_dim).reshape(batch * seq_len * heads, half_dim)
        sin_theta = sin_theta.unsqueeze(2).expand(batch, seq_len, heads, half_dim).reshape(batch * seq_len * heads, half_dim)
    else:
        cos_theta = cos_theta.reshape(batch * seq_len, half_dim)
        sin_theta = sin_theta.reshape(batch * seq_len, half_dim)

    # Rotate
    rotated_even = x_even * cos_theta - x_odd * sin_theta
    rotated_odd = x_even * sin_theta + x_odd * cos_theta

    # Interleave back
    x_rotated = torch.stack([rotated_even, rotated_odd], dim=-1)
    x_rotated = x_rotated.reshape(-1, dim)

    # Reshape to original dimensions
    if heads is not None:
        x_rotated = x_rotated.view(batch, seq_len, heads, dim)
    else:
        x_rotated = x_rotated.view(batch, seq_len, dim)

    return x_rotated

class TextRotaryEmbedding(nn.Module):
    """Rotary embedding generator """

    def __init__(
        self,
        config: TextConfig,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        head_dim = config.d_rh  # Use d_rh for rotary dimension
        assert head_dim % 2 == 0, "Rotary dimension must be even for RoPE."

        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len_cached = 0
        self.freqs_theta_cache: Optional[torch.Tensor] = None
        self.attention_scaling = 1.0

    def _build_cache(self, seq_len: int, device: torch.device) -> torch.Tensor:
        positions = torch.arange(seq_len, device=device).float()
        # Outer product: [seq_len, head_dim//2]
        return torch.outer(positions, self.inv_freq)

    def forward(
        self,
        position_ids: torch.LongTensor,
        cache_position: Optional[torch.LongTensor] = None
    ) -> torch.Tensor:
        positions = cache_position if cache_position is not None else position_ids
        max_position = positions.max().item() + 1  # +1 because indexing is exclusive

        device = positions.device
        if self.freqs_theta_cache is None or max_position > self.max_seq_len_cached:
            self.freqs_theta_cache = self._build_cache(max_position, device)
            self.max_seq_len_cached = max_position

        if positions.ndim == 2:
            theta = self.freqs_theta_cache[positions]
        else:
            theta = self.freqs_theta_cache[positions]

        return theta * self.attention_scaling