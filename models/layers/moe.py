import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from configs.model_config import TextConfig
from models.layers.mlp import MLP


class Moe(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        # Number of experts each token is routed to (top-K)
        self.num_experts_per_token = config.num_experts_per_tok
        # Total number of local experts available
        self.total_local_experts = config.num_local_experts
        # Number of shared experts always applied
        self.num_shared_experts = 2
        # Hidden dimension size
        self.hidden_size = config.hidden_size

        # Linear router: maps each token to scores for each local expert
        self.expert_router = nn.Linear(self.hidden_size, self.total_local_experts)

        # Local experts (only top-K selected per token)
        self.local_experts = nn.ModuleList([MLP(config) for _ in range(self.total_local_experts)])
        # Shared experts (applied to every token)
        self.shared_experts = nn.ModuleList([MLP(config) for _ in range(self.num_shared_experts)])

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: Tensor of shape [batch_size, seq_len, hidden_size]
        Returns:
            output: Tensor of shape [batch_size, seq_len, hidden_size]
            local_expert_scores: Raw scores from router of shape [batch_size, seq_len, total_local_experts]
        """
        batch_size, seq_len, _ = hidden_states.shape
        # Flatten tokens for single-dimension expert calls: [batch_size * seq_len, hidden_size]
        flattened_hidden_states = hidden_states.view(-1, self.hidden_size)

        # 1) Compute outputs from shared experts applied to all tokens
        shared_experts_output = self._aggregate_shared_experts(flattened_hidden_states)

        # 2) Compute routing probabilities for local experts
        local_expert_scores = self.expert_router(hidden_states)  # [batch_size, seq_len, total_local_experts]
        local_expert_probabilities = F.softmax(local_expert_scores, dim=-1)
        top_k_probabilities, top_k_expert_indices = local_expert_probabilities.topk(self.num_experts_per_token, dim=-1)
        # top_k_probabilities, top_k_expert_indices: [batch_size, seq_len, num_experts_per_token]

        # 3) Route tokens to their selected experts and apply them
        routed_experts_output = self._route_and_apply_local_experts(
            flattened_hidden_states, top_k_expert_indices, top_k_probabilities
        )

        # 4) Combine residual, shared, and routed contributions
        combined_output_flat = flattened_hidden_states + shared_experts_output + routed_experts_output
        output = combined_output_flat.view(batch_size, seq_len, self.hidden_size)
        return output, local_expert_scores

    def _aggregate_shared_experts(self, flattened_hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Runs all shared experts on every token and sums their outputs.
        Args:
            flattened_hidden_states: [batch_size * seq_len, hidden_size]
        Returns:
            shared_experts_sum: [batch_size * seq_len, hidden_size]
        """
        shared_experts_sum = torch.zeros_like(flattened_hidden_states)
        for expert in self.shared_experts:
            shared_experts_sum += expert(flattened_hidden_states)
        return shared_experts_sum

    def _route_and_apply_local_experts(
        self,
        flattened_hidden_states: torch.Tensor,
        top_k_expert_indices: torch.Tensor,
        top_k_probabilities: torch.Tensor
    ) -> torch.Tensor:
        """
        Routes each token to its top-K local experts, applies those experts,
        then scatters weighted outputs back to the original token positions.
        Args:
            flattened_hidden_states: [batch_size * seq_len, hidden_size]
            top_k_expert_indices: [batch_size, seq_len, num_experts_per_token]
            top_k_probabilities: [batch_size, seq_len, num_experts_per_token]
        Returns:
            routed_experts_output: [batch_size * seq_len, hidden_size]
        """
        num_tokens, hidden_size = flattened_hidden_states.shape
        K = self.num_experts_per_token
        E = self.total_local_experts

        # Flatten batch/time dims: [batch_size * seq_len, num_experts_per_token]
        flattened_top_k_indices = top_k_expert_indices.view(-1, K)
        flattened_top_k_probabilities = top_k_probabilities.view(-1, K)

        routed_experts_output = torch.zeros_like(flattened_hidden_states)

        # Process each expert separately
        for expert_id, expert in enumerate(self.local_experts):
            # Mask to identify tokens routed to this expert
            tokens_for_expert_mask = (flattened_top_k_indices == expert_id)  # [batch_size * seq_len, K]
            tokens_using_expert = tokens_for_expert_mask.any(dim=1)  # [batch_size * seq_len]

            if not tokens_using_expert.any():
                continue  # this expert got no tokens

            # Indices of tokens for this expert
            selected_token_indices = tokens_using_expert.nonzero(as_tuple=False).squeeze(1)
            # Compute each token's total routing weight for this expert
            expert_weights_for_tokens = (flattened_top_k_probabilities[selected_token_indices] * tokens_for_expert_mask[selected_token_indices]).sum(dim=1, keepdim=True)  # [N,1]

            # Apply the expert to the selected tokens
            selected_token_inputs = flattened_hidden_states[selected_token_indices]  # [N, hidden_size]
            expert_outputs_for_tokens = expert(selected_token_inputs)  # [N, hidden_size]

            # Scatter weighted outputs back to full sequence
            routed_experts_output[selected_token_indices] += expert_outputs_for_tokens * expert_weights_for_tokens

        return routed_experts_output