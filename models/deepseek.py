# models/deepseek.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List, Dict
from configs.model_config import TextConfig
from models.layers.rotary import TextRotaryEmbedding
from models.layers.norm import RMSNorm
from models.decoder_layer import DecoderLayer
import logging

class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.n_layers = config.n_layers
        self.max_seq_len = config.max_seq_len
        self.padding_idx = config.pad_token_id

        self.token_embed = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.hidden_size,
            padding_idx=self.padding_idx
        )
        self.rotary_emb = TextRotaryEmbedding(config)
        self.layers = nn.ModuleList([
            DecoderLayer(config, layer_idx=i) for i in range(self.n_layers)
        ])
        self.final_norm = RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)

        self.register_buffer(
            "pos_id_buffer",
            torch.arange(self.max_seq_len, dtype=torch.long),
            persistent=False
        )

    def _create_causal_mask(self, seq_len: int, device: torch.device, past_length: int = 0) -> torch.Tensor:
        total_len = seq_len + past_length
        mask = torch.triu(torch.full((seq_len, total_len), float('-inf'), device=device), diagonal=1 + past_length)
        return mask.unsqueeze(0).unsqueeze(0)  # shape: (1,1,seq_len,total_len)

    def forward(
        self,
        input_ids: torch.Tensor,  # (B, T)
        attention_mask: Optional[torch.Tensor] = None,  # (B, T)
        position_ids: Optional[torch.Tensor] = None,  # (B, T)
        past_key_values: Optional[List[Dict[str, torch.Tensor]]] = None,  # length = n_layers
        output_attentions: bool = False,
        output_router_logits: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]], Optional[List[torch.Tensor]], Optional[List[Dict[str, torch.Tensor]]]]:
        B, T = input_ids.size()
        device = input_ids.device

        # Embed tokens
        hidden_states = self.token_embed(input_ids)  # (B, T, hidden_size)

        # Prepare position_ids if not given
        if position_ids is None:
            if use_cache and past_key_values is not None and any(pkv is not None for pkv in past_key_values):
                current_pos = past_key_values[0]["compressed_kv"].shape[1] if "compressed_kv" in past_key_values[0] else 0
                position_ids = torch.arange(current_pos, current_pos + T, device=device).unsqueeze(0).expand(B, -1)
            else:
                current_pos = 0
                position_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        else:
            current_pos = position_ids[0, 0].item()

        # Rotary embedding on positions
        theta = self.rotary_emb(position_ids)  # (B, T, hidden_size or something)

        # Prepare causal mask
        past_length = current_pos if use_cache and past_key_values is not None else 0
        causal_mask = self._create_causal_mask(T, device, past_length)

        # Combine attention_mask and causal_mask if attention_mask exists
        if attention_mask is not None:
            if use_cache and past_key_values is not None:
                pad_len = past_length
                attention_mask = torch.cat(
                    [torch.ones(B, pad_len, device=device), attention_mask], dim=1
                )
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # broadcast to (B,1,1,S)
            attention_mask = attention_mask + causal_mask  # broadcasting add (mask is -inf where masked)
        else:
            attention_mask = causal_mask  # (1,1,T,total_len)

        # Prepare outputs containers
        all_attentions = [] if output_attentions else None
        all_router_logits = [] if output_router_logits else None
        next_past_key_values = [] if use_cache else None

        # Iterate over decoder layers
        for i, layer in enumerate(self.layers):
            layer_past = past_key_values[i] if (past_key_values is not None and len(past_key_values) > i) else None

            hidden_states, attn_weights, router_logits, new_past = layer(
                hidden_states=hidden_states,
                position_embeddings=theta,
                attention_mask=attention_mask,
                past_key_value=layer_past,
                output_attentions=output_attentions,
                output_router_logits=output_router_logits,
                use_cache=use_cache
            )

            if output_attentions:
                all_attentions.append(attn_weights)
            if output_router_logits:
                all_router_logits.append(router_logits)
            if use_cache:
                next_past_key_values.append(new_past)

        # Final norm + LM head
        hidden_states = self.final_norm(hidden_states)
        logits = self.lm_head(hidden_states)

        return logits, all_attentions, all_router_logits, next_past_key_values
