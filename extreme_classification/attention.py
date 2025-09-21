"""
Tail-Adaptive Attention Mechanism

This module implements a specialized attention mechanism designed to improve
the detection and classification of rare/tail classes in extreme multi-label
classification tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class TailAdaptiveAttention(nn.Module):
    """
    Tail-Adaptive Attention mechanism for better rare class detection.
    
    This attention mechanism adaptively focuses on features that are most 
    discriminative for tail (rare) labels, helping to improve their detection
    in extreme multi-label classification scenarios.
    
    Args:
        hidden_size: Size of hidden representations
        num_labels: Total number of labels
        tail_threshold: Threshold below which labels are considered tail
        num_attention_heads: Number of attention heads
        attention_dropout: Dropout rate for attention weights
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_labels: int,
        tail_threshold: float = 0.01,
        num_attention_heads: int = 8,
        attention_dropout: float = 0.1
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.tail_threshold = tail_threshold
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        # Query, Key, Value projections for tail-aware attention
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        
        # Tail-specific transformations
        self.tail_query_transform = nn.Linear(hidden_size, self.all_head_size)
        self.tail_key_transform = nn.Linear(hidden_size, self.all_head_size)
        
        # Attention output projection
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(attention_dropout)
        
        # Tail-adaptive gating mechanism
        self.tail_gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()
        )
        
        # Label-specific attention weights
        self.label_embeddings = nn.Embedding(num_labels, hidden_size)
        self.label_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=4,
            dropout=attention_dropout,
            batch_first=True
        )
        
        # Frequency-based weighting
        self.frequency_transform = nn.Sequential(
            nn.Linear(1, hidden_size // 8),
            nn.ReLU(),
            nn.Linear(hidden_size // 8, 1)
        )
        
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Transpose tensor for multi-head attention computation."""
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def compute_tail_attention_weights(
        self,
        sequence_output: torch.Tensor,
        tail_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute attention weights that focus on tail label features.
        
        Args:
            sequence_output: Sequence representations [batch_size, seq_len, hidden_size]
            tail_mask: Boolean mask indicating tail labels [num_labels]
            
        Returns:
            Attention weights for tail-focused features
        """
        batch_size, seq_len, hidden_size = sequence_output.shape
        
        # Get tail label embeddings
        tail_indices = torch.where(tail_mask)[0]
        if len(tail_indices) == 0:
            # No tail labels, return uniform attention
            return torch.ones(batch_size, seq_len, device=sequence_output.device) / seq_len
        
        tail_embeddings = self.label_embeddings(tail_indices)  # [num_tail_labels, hidden_size]
        
        # Compute attention between sequence tokens and tail label embeddings
        sequence_flat = sequence_output.view(-1, hidden_size)  # [batch_size * seq_len, hidden_size]
        
        # Attention scores between tokens and tail embeddings
        attention_scores = torch.matmul(
            sequence_flat, tail_embeddings.transpose(0, 1)
        )  # [batch_size * seq_len, num_tail_labels]
        
        # Aggregate attention scores across tail labels
        tail_attention = attention_scores.max(dim=-1)[0]  # [batch_size * seq_len]
        tail_attention = tail_attention.view(batch_size, seq_len)
        
        # Apply softmax to get attention weights
        tail_attention_weights = F.softmax(tail_attention, dim=-1)
        
        return tail_attention_weights
    
    def forward(
        self,
        sequence_output: torch.Tensor,
        pooled_output: torch.Tensor,
        tail_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of tail-adaptive attention.
        
        Args:
            sequence_output: Sequence representations [batch_size, seq_len, hidden_size]
            pooled_output: Pooled representation [batch_size, hidden_size]
            tail_mask: Boolean mask for tail labels [num_labels]
            attention_mask: Attention mask for input sequences [batch_size, seq_len]
            
        Returns:
            Tail-adapted attention features [batch_size, hidden_size]
        """
        batch_size, seq_len, hidden_size = sequence_output.shape
        
        # Standard multi-head attention
        mixed_query_layer = self.query(sequence_output)
        mixed_key_layer = self.key(sequence_output)
        mixed_value_layer = self.value(sequence_output)
        
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        
        # Compute attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attention_mask_expanded = attention_mask[:, None, None, :].expand_as(attention_scores)
            attention_scores = attention_scores.masked_fill(
                attention_mask_expanded == 0, -10000.0
            )
        
        # Standard attention weights
        standard_attention_probs = F.softmax(attention_scores, dim=-1)
        standard_attention_probs = self.dropout(standard_attention_probs)
        
        # Apply standard attention
        standard_context = torch.matmul(standard_attention_probs, value_layer)
        standard_context = standard_context.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = standard_context.size()[:-2] + (self.all_head_size,)
        standard_context = standard_context.view(*new_context_layer_shape)
        
        # Tail-specific attention if tail mask is provided
        if tail_mask is not None and tail_mask.any():
            # Compute tail-focused attention weights
            tail_attention_weights = self.compute_tail_attention_weights(
                sequence_output, tail_mask
            )
            
            # Apply tail attention to get tail-focused features
            tail_weighted_sequence = sequence_output * tail_attention_weights.unsqueeze(-1)
            tail_features = tail_weighted_sequence.mean(dim=1)  # [batch_size, hidden_size]
            
            # Tail-specific transformations
            tail_query = self.tail_query_transform(pooled_output)
            tail_key = self.tail_key_transform(tail_features)
            
            # Compute tail attention score
            tail_attention_score = torch.sum(tail_query * tail_key, dim=-1, keepdim=True)
            tail_gate_weight = self.tail_gate(pooled_output)
            
            # Combine standard and tail features
            combined_features = (
                (1 - tail_gate_weight) * standard_context.mean(dim=1) +
                tail_gate_weight * tail_features
            )
        else:
            # No tail labels, use standard attention
            combined_features = standard_context.mean(dim=1)
        
        # Final transformation
        attention_output = self.dense(combined_features)
        attention_output = self.dropout(attention_output)
        attention_output = self.LayerNorm(attention_output + pooled_output)
        
        return attention_output


class AdaptivePositionalAttention(nn.Module):
    """
    Adaptive positional attention that adjusts based on label frequency.
    
    This module provides position-aware attention that gives different weights
    to different positions in the sequence based on label characteristics.
    """
    
    def __init__(self, hidden_size: int, max_position: int = 512):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_position = max_position
        
        # Learnable positional embeddings
        self.position_embeddings = nn.Embedding(max_position, hidden_size)
        
        # Attention mechanism for positions
        self.position_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Frequency-adaptive transformation
        self.frequency_adapter = nn.Sequential(
            nn.Linear(hidden_size + 1, hidden_size),  # +1 for frequency info
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
    def forward(
        self,
        sequence_output: torch.Tensor,
        label_frequencies: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply adaptive positional attention.
        
        Args:
            sequence_output: Input sequences [batch_size, seq_len, hidden_size]
            label_frequencies: Label frequency information [batch_size, num_labels]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Position-adapted features [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, hidden_size = sequence_output.shape
        
        # Get positional embeddings
        positions = torch.arange(seq_len, device=sequence_output.device)
        position_embeddings = self.position_embeddings(positions)
        position_embeddings = position_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Combine sequence output with positional information
        combined_input = sequence_output + position_embeddings
        
        # Apply positional attention
        attended_output, attention_weights = self.position_attention(
            query=combined_input,
            key=combined_input,
            value=sequence_output,
            key_padding_mask=attention_mask == 0 if attention_mask is not None else None
        )
        
        # Adapt based on label frequencies if provided
        if label_frequencies is not None:
            # Use average frequency as a signal
            avg_frequency = label_frequencies.mean(dim=-1, keepdim=True)  # [batch_size, 1]
            
            # Expand frequency info to match sequence length
            freq_expanded = avg_frequency.unsqueeze(1).expand(-1, seq_len, -1)
            
            # Combine with attention output
            combined_features = torch.cat([attended_output, freq_expanded], dim=-1)
            adapted_output = self.frequency_adapter(combined_features)
        else:
            adapted_output = attended_output
        
        return adapted_output