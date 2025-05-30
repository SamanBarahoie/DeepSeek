from typing import Optional, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from models.layers.rotary import TextRotaryEmbedding, apply_rope_matrix
from configs.model_config import TextConfig

class LoRALinear(nn.Module):
    """
    A linear layer with Low-Rank Adaptation (LoRA).
    Adds low-rank matrices A and B to the frozen weight matrix.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        lora_rank: int,
        lora_alpha: float = 1.0,
        bias: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha

        # Original linear layer (frozen)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.weight.requires_grad = True  # Freeze original weights

        # LoRA matrices
        self.lora_A = nn.Parameter(torch.empty(lora_rank, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, lora_rank))
        
        # Initialize LoRA matrices
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)  # Zero-init B for stable starting point

        # Bias (optional, matching original nn.Linear)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

        # Scaling factor
        self.lora_scaling = lora_alpha / lora_rank if lora_rank > 0 else 1.0

        # Flag to enable/disable LoRA
        self.lora_enabled = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base linear transformation
        output = F.linear(x, self.weight, self.bias)

        # Add LoRA contribution if enabled
        if self.lora_enabled and self.lora_rank > 0:
            lora_output = F.linear(F.linear(x, self.lora_A), self.lora_B)
            output = output + lora_output * self.lora_scaling

        return output

class MultiHeadLatentAttention(nn.Module):
    """
    Implements a multi-head attention mechanism with low-rank latent projection for keys and values,
    combined with decoupled Rotary Position Embeddings (RoPE) for efficiency and memory.
    Includes LoRA for fine-tuning with methods to enable/disable LoRA.
    """

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        compress_dim_kv: int,
        compress_dim_q: int,
        rope_dim: int,
        config: TextConfig,
        lora_rank: int = 16,  # LoRA rank, configurable
        lora_alpha: float = 32.0,  # LoRA scaling factor
    ):
        super().__init__()
        if model_dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")

        # Core dimensions
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.compress_dim_kv = compress_dim_kv
        self.compress_dim_q = compress_dim_q
        self.rope_dim = rope_dim
        self.attention_dim = self.head_dim + rope_dim
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha

        # Low-rank compression projections for keys and values with LoRA
        self.kv_compress = LoRALinear(model_dim, compress_dim_kv, lora_rank, lora_alpha, bias=False)
        self.kv_uncompress_key = LoRALinear(compress_dim_kv, model_dim, lora_rank, lora_alpha, bias=False)
        self.kv_uncompress_value = LoRALinear(compress_dim_kv, model_dim, lora_rank, lora_alpha, bias=False)

        # Low-rank compression for queries with LoRA
        self.q_compress = LoRALinear(model_dim, compress_dim_q, lora_rank, lora_alpha, bias=False)
        self.q_uncompress = LoRALinear(compress_dim_q, model_dim, lora_rank, lora_alpha, bias=False)

        # Projections for decoupled RoPE components with LoRA
        self.key_rope_projector = LoRALinear(model_dim, rope_dim, lora_rank, lora_alpha, bias=False)
        self.query_rope_projector = LoRALinear(compress_dim_q, num_heads * rope_dim, lora_rank, lora_alpha, bias=False)

        # Final output projection combining all heads with LoRA
        self.output_projection = LoRALinear(num_heads * self.head_dim, model_dim, lora_rank, lora_alpha, bias=False)

        # RoPE embedding utility
        self.rotary_embedding = TextRotaryEmbedding(config)

    def enable_lora(self):
        """Enable LoRA for all LoRALinear layers in the module."""
        for module in self.modules():
            if isinstance(module, LoRALinear):
                module.lora_enabled = True

    def disable_lora(self):
        """Disable LoRA for all LoRALinear layers in the module."""
        for module in self.modules():
            if isinstance(module, LoRALinear):
                module.lora_enabled = False

    def get_lora_state(self) -> bool:
        """Check if LoRA is enabled for the module (returns True if all LoRALinear layers have LoRA enabled)."""
        lora_states = [module.lora_enabled for module in self.modules() if isinstance(module, LoRALinear)]
        return all(lora_states) if lora_states else False

    def _process_keys_values(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor,
        past_states: Optional[Dict[str, torch.Tensor]],
        use_cache: bool,
        batch_size: int,
        time_steps: int
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Process keys and values: compress, apply RoPE, handle caching, and prepare for multi-head attention.
        Returns:
            - Full keys and values shaped for multi-head attention.
            - Updated cache if use_cache is True.
        """
        # Compress key/value representations for the current step
        kv_compressed_current = self.kv_compress(hidden_states)  # Shape: [batch_size, time_steps, compress_dim_kv]

        # Create position-aware key representations using RoPE
        key_rope_current = self.key_rope_projector(hidden_states)  # Shape: [batch_size, time_steps, rope_dim]
        key_rope_current = apply_rope_matrix(key_rope_current, position_embeddings)  # Apply position embeddings

        # Handle caching for incremental decoding (e.g., during generation)
        if use_cache:
            if past_states is None:
                # Initialize cache with current step's data
                past_states = {
                    "compressed_kv": kv_compressed_current,
                    "key_rope": key_rope_current,
                }
            else:
                # Append current step's data to past cache
                past_states["compressed_kv"] = torch.cat(
                    [past_states["compressed_kv"], kv_compressed_current], dim=1
                )
                past_states["key_rope"] = torch.cat(
                    [past_states["key_rope"], key_rope_current], dim=1
                )
            all_compressed_kv = past_states["compressed_kv"]
            all_key_rope = past_states["key_rope"]
        else:
            # No caching: use only current step's data
            all_compressed_kv = kv_compressed_current
            all_key_rope = key_rope_current

        # Uncompress to full key and value representations
        full_keys = self.kv_uncompress_key(all_compressed_kv)  # Shape: [batch_size, total_time_steps, model_dim]
        full_values = self.kv_uncompress_value(all_compressed_kv)  # Shape: [batch_size, total_time_steps, model_dim]

        # Reshape for multi-head attention: [batch_size, total_time_steps, num_heads, head_dim]
        full_keys = full_keys.view(batch_size, -1, self.num_heads, self.head_dim)
        full_values = full_values.view(batch_size, -1, self.num_heads, self.head_dim)

        # Expand RoPE keys for each head: [batch_size, total_time_steps, num_heads, rope_dim]
        rope_keys_per_head = all_key_rope.unsqueeze(2).expand(-1, -1, self.num_heads, -1)

        return full_keys, full_values, rope_keys_per_head, past_states

    def _process_queries(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor,
        batch_size: int,
        time_steps: int
    ) -> torch.Tensor:
        """
        Process queries: compress, uncompress, apply RoPE, and prepare for attention.
        Returns:
            - Combined query tensor with content and position information.
        """
        # Compress queries for the current step
        query_compressed = self.q_compress(hidden_states)  # Shape: [batch_size, time_steps, compress_dim_q]

        # Uncompress to full query representations
        query_content = self.q_uncompress(query_compressed)  # Shape: [batch_size, time_steps, model_dim]

        # Reshape for multi-head attention: [batch_size, time_steps, num_heads, head_dim]
        query_content = query_content.view(batch_size, time_steps, self.num_heads, self.head_dim)

        # Project queries for RoPE and apply position embeddings
        query_rope = self.query_rope_projector(query_compressed)  # Shape: [batch_size, time_steps, num_heads * rope_dim]
        query_rope = query_rope.view(batch_size, time_steps, self.num_heads, self.rope_dim)
        query_rope = apply_rope_matrix(query_rope, position_embeddings)  # Shape: [batch_size, time_steps, num_heads, rope_dim]

        # Combine content and position-aware queries
        query_combined = torch.cat([query_content, query_rope], dim=-1)  # Shape: [batch_size, time_steps, num_heads, head_dim + rope_dim]

        return query_combined

    def _compute_attention(
        self,
        query_combined: torch.Tensor,
        full_keys: torch.Tensor,
        rope_keys_per_head: torch.Tensor,
        full_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        batch_size: int
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute scaled dot-product attention scores and apply to values.
        Returns:
            - Attention output per head.
            - Attention weights (if requested).
        """
        # Combine keys with RoPE keys
        key_combined = torch.cat([full_keys, rope_keys_per_head], dim=-1)  # Shape: [batch_size, total_time_steps, num_heads, head_dim + rope_dim]

        # Reshape for batched matrix multiplication
        query_for_matmul = query_combined.permute(0, 2, 1, 3)  # Shape: [batch_size, num_heads, time_steps, head_dim + rope_dim]
        key_for_matmul = key_combined.permute(0, 2, 3, 1)  # Shape: [batch_size, num_heads, head_dim + rope_dim, total_time_steps]

        # Compute attention scores
        scale = 1.0 / math.sqrt(self.attention_dim)
        attention_scores = torch.matmul(query_for_matmul, key_for_matmul) * scale  # Shape: [batch_size, num_heads, time_steps, total_time_steps]

        # Apply attention mask if provided
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Compute attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Apply attention weights to values
        value_for_matmul = full_values.permute(0, 2, 1, 3)  # Shape: [batch_size, num_heads, total_time_steps, head_dim]
        attention_output_per_head = torch.matmul(attention_weights, value_for_matmul)  # Shape: [batch_size, num_heads, time_steps, head_dim]

        return attention_output_per_head, attention_weights

    def _finalize_output(
        self,
        attention_output_per_head: torch.Tensor,
        batch_size: int,
        time_steps: int
    ) -> torch.Tensor:
        """
        Reshape and project attention output to final dimensions.
        Returns:
            - Final attention output.
        """
        # Reshape attention output: [batch_size, time_steps, num_heads, head_dim] -> [batch_size, time_steps, num_heads * head_dim]
        attention_output = attention_output_per_head.permute(0, 2, 1, 3).contiguous().view(batch_size, time_steps, -1)

        # Project to final output dimension
        final_output = self.output_projection(attention_output)  # Shape: [batch_size, time_steps, model_dim]

        return final_output

    def forward(
        self,
        hidden_states: torch.Tensor,                        # [batch_size, time_steps, model_dim]
        position_embeddings: torch.Tensor,                   # [batch_size, time_steps, rope_dim]
        attention_mask: Optional[torch.Tensor] = None,
        past_states: Optional[Dict[str, torch.Tensor]] = None,
        output_attention_weights: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Dict[str, torch.Tensor]]]:
        """
        Compute multi-head attention with low-rank compression and RoPE.
        
        Args:
            hidden_states: Input tensor containing the data to process [batch_size, time_steps, model_dim].
            position_embeddings: Positional information for RoPE [batch_size, time_steps, rope_dim//2].
            attention_mask: Optional tensor to mask attention scores (e.g., for padding or causal attention).
            past_states: Optional dictionary with cached keys and values for incremental decoding.
            output_attention_weights: If True, return the attention weights.
            use_cache: If True, cache keys and values for future steps.

        Returns:
            - Final output tensor after attention [batch_size, time_steps, model_dim].
            - Attention weights if output_attention_weights is True [batch_size, num_heads, time_steps, total_time_steps].
            - Updated cache if use_cache is True.
        """
        # Get batch size and time steps from input
        batch_size, time_steps, _ = hidden_states.size()

        # Step 1: Process keys and values (compress, apply RoPE, handle caching)
        full_keys, full_values, rope_keys_per_head, updated_cache = self._process_keys_values(
            hidden_states, position_embeddings, past_states, use_cache, batch_size, time_steps
        )

        # Step 2: Process queries (compress, uncompress, apply RoPE)
        query_combined = self._process_queries(hidden_states, position_embeddings, batch_size, time_steps)

        # Step 3: Compute attention scores and apply to values
        attention_output_per_head, attention_weights = self._compute_attention(
            query_combined, full_keys, rope_keys_per_head, full_values, attention_mask, batch_size
        )

        # Step 4: Reshape and project to final output
        final_output = self._finalize_output(attention_output_per_head, batch_size, time_steps)

        # Prepare outputs: include attention weights if requested, cache if used
        return (
            final_output,
            attention_weights if output_attention_weights else None,
            updated_cache if use_cache else None
        )