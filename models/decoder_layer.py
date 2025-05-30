import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict
from configs.model_config import TextConfig
from models.layers.MLA import MultiHeadLatentAttention
from models.layers.moe import Moe
from models.layers.mlp import MLP
from models.layers.norm import RMSNorm
import logging

# Configure logging for debugging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class DecoderLayer(nn.Module):
    """A single decoder layer for a transformer model, combining attention and feed-forward processing."""
    
    def __init__(self, config: TextConfig, layer_idx: int):
        """
        Initializes the decoder layer with attention and feed-forward components.
        
        Args:
            config: Configuration object containing model hyperparameters (e.g., hidden_size, num_attention_heads).
            layer_idx: Index of this layer in the transformer stack.

        Raises:
            ValueError: If config parameters are invalid (e.g., hidden_size not divisible by num_attention_heads).
        """
        super().__init__()
        self.layer_idx = layer_idx  # Store layer index for debugging/logging
        
        # Validate configuration
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({config.hidden_size}) must be divisible by "
                f"num_attention_heads ({config.num_attention_heads})"
            )

        # Initialize multi-head latent attention
        self.attention = MultiHeadLatentAttention(
            model_dim=config.hidden_size,  # Model dimension for attention
            num_heads=config.num_attention_heads,
            compress_dim_kv=config.d_c,
            compress_dim_q=config.d_c_q,
            rope_dim=config.d_rh,
            config=config,
            lora_rank=config.lora_rank,  # LoRA rank from config
            lora_alpha=config.lora_alpha,  # LoRA scaling factor from config
        )
        
        # Initialize feed-forward layer (MoE or MLP based on layer_idx)
        self.feed_forward = Moe(config) if layer_idx in config.moe_layers else MLP(config)
        
        # Initialize normalization layers
        self.norm1 = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm2 = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        logging.info(
            f"DecoderLayer {layer_idx} initialized: hidden_size={config.hidden_size}, "
            f"num_attention_heads={config.num_attention_heads}, feed_forward_type={type(self.feed_forward).__name__}, "
            f"lora_rank={config.lora_rank}, lora_alpha={config.lora_alpha}"
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        chunk_causal_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Dict] = None,
        output_attentions: bool = False,
        output_router_logits: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[Dict]]:
        """
        Forward pass of the decoder layer, applying attention, normalization, and feed-forward processing.
        
        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size].
            position_embeddings: Positional encodings for the input sequence.
            attention_mask: Optional mask for attention, shape [batch_size, 1, seq_len, seq_len].
            chunk_causal_mask: Optional causal mask for chunked attention.
            past_key_value: Optional dictionary of cached key/value tensors for fast decoding.
            output_attentions: If True, returns attention weights.
            output_router_logits: If True, returns MoE router logits (if applicable).
            use_cache: If True, returns updated key/value cache for fast decoding.
        
        Returns:
            Tuple containing:
            - hidden_states: Output tensor of shape [batch_size, seq_len, hidden_size].
            - attn_weights: Optional attention weights if output_attentions is True.
            - router_logits: Optional MoE router logits if output_router_logits is True and feed-forward is MoE.
            - past_key_value: Updated key/value cache if use_cache is True.
        """
        # Store input for residual connection
        residual = hidden_states
        
        # Apply first normalization before attention
        hidden_states = self.norm1(hidden_states)
        logging.debug(
            f"Layer {self.layer_idx} after norm1: hidden_states shape: {hidden_states.shape}, "
            f"requires_grad: {hidden_states.requires_grad}"
        )

        # Apply multi-head latent attention
        attention_output, attn_weights, past_key_value = self.attention(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_states=past_key_value,
            output_attention_weights=output_attentions,
            use_cache=use_cache
        )
        logging.debug(
            f"Layer {self.layer_idx} after attention: attention_output shape: {attention_output.shape}, "
            f"requires_grad: {attention_output.requires_grad}"
        )

        # Add residual connection after attention
        hidden_states = residual + attention_output
        residual = hidden_states

        # Apply second normalization before feed-forward
        hidden_states = self.norm2(hidden_states)
        logging.debug(
            f"Layer {self.layer_idx} after norm2: hidden_states shape: {hidden_states.shape}, "
            f"requires_grad: {hidden_states.requires_grad}"
        )

        # Apply feed-forward layer (MoE or MLP)
        if isinstance(self.feed_forward, Moe):
            ff_output, router_logits = self.feed_forward(hidden_states)
            logging.debug(
                f"Layer {self.layer_idx} after MoE: ff_output shape: {ff_output.shape}, "
                f"requires_grad: {ff_output.requires_grad}, "
                f"router_logits shape: {router_logits.shape}, requires_grad: {router_logits.requires_grad}"
            )
        else:
            ff_output = self.feed_forward(hidden_states)
            router_logits = None
            logging.debug(
                f"Layer {self.layer_idx} after MLP: ff_output shape: {ff_output.shape}, "
                f"requires_grad: {ff_output.requires_grad}"
            )

        # Add residual connection after feed-forward
        hidden_states = residual + ff_output

        # Return outputs in the specified order
        return (
            hidden_states,
            attn_weights,
            router_logits if output_router_logits and isinstance(self.feed_forward, Moe) else None,
            past_key_value
        )