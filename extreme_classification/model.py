"""
Enhanced ExectModel for Extreme Multi-Label Classification

This module implements an improved version of ExectModel that focuses on enhancing
Precision@3 and NDCG@5 metrics through better tail label handling and specialized
attention mechanisms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from .attention import TailAdaptiveAttention
from .loss import FocusedLoss, ContrastiveLoss
from .calibration import TopKCalibrator


class ExectModel(nn.Module):
    """
    Enhanced ExectModel for extreme multi-label classification.
    
    This model incorporates several improvements over standard BERT-based models:
    1. TailAdaptiveAttention for better rare class detection
    2. Enhanced architecture for tail label handling
    3. Multi-scale feature extraction
    4. Calibrated predictions for top-k accuracy
    
    Args:
        config: Model configuration
        num_labels: Number of output labels
        bert_model_name: Name of the BERT model to use as backbone
        tail_threshold: Frequency threshold below which labels are considered "tail"
        use_tail_attention: Whether to use tail-adaptive attention
        dropout_rate: Dropout rate for regularization
    """
    
    def __init__(
        self,
        config: Optional[BertConfig] = None,
        num_labels: int = 10000,
        bert_model_name: str = "hfl/chinese-macbert-base",
        tail_threshold: float = 0.01,
        use_tail_attention: bool = True,
        dropout_rate: float = 0.1,
        temperature: float = 1.0,
        **kwargs
    ):
        super().__init__()
        
        self.num_labels = num_labels
        self.tail_threshold = tail_threshold
        self.use_tail_attention = use_tail_attention
        self.temperature = temperature
        
        # Initialize BERT backbone
        if config is None:
            if bert_model_name:
                self.bert = BertModel.from_pretrained(bert_model_name)
                config = self.bert.config
                self.hidden_size = config.hidden_size
            else:
                # Use default config for testing
                self.hidden_size = 768  # Default hidden size
                config = BertConfig(hidden_size=self.hidden_size)
                self.bert = BertModel(config)
        else:
            self.bert = BertModel(config)
            self.hidden_size = config.hidden_size
        
        # Multi-scale feature extraction
        self.feature_extractors = nn.ModuleList([
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.Linear(self.hidden_size, self.hidden_size // 4)
        ])
        
        # Tail-adaptive attention mechanism
        if self.use_tail_attention:
            self.tail_attention = TailAdaptiveAttention(
                hidden_size=self.hidden_size,
                num_labels=num_labels,
                tail_threshold=tail_threshold
            )
        
        # Enhanced classifier with multiple heads
        self.dropout = nn.Dropout(dropout_rate)
        
        # Main classifier
        classifier_input_size = self.hidden_size
        if self.use_tail_attention:
            classifier_input_size += self.hidden_size  # Add tail attention features
            
        # Multi-head classification
        self.head_classifier = nn.Linear(classifier_input_size, num_labels)
        self.tail_classifier = nn.Linear(classifier_input_size, num_labels)
        
        # Feature fusion layer
        fusion_size = self.hidden_size + self.hidden_size // 2 + self.hidden_size // 4
        self.feature_fusion = nn.Sequential(
            nn.Linear(fusion_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.hidden_size, self.hidden_size)
        )
        
        # Calibration layer for top-k predictions
        self.calibrator = TopKCalibrator(num_labels, k_values=[1, 3, 5])
        
        # Loss functions
        self.focused_loss = FocusedLoss(alpha=2.0, gamma=2.0)
        self.contrastive_loss = ContrastiveLoss(temperature=0.1)
        
        # Label frequency tracking (for identifying tail labels)
        self.register_buffer('label_frequencies', torch.zeros(num_labels))
        self.register_buffer('update_count', torch.tensor(0))
        
    def update_label_frequencies(self, labels: torch.Tensor):
        """Update label frequency statistics for tail label identification."""
        if labels.dim() == 2:  # Multi-hot encoding
            freq_update = labels.sum(dim=0).float()
        else:  # Label indices
            freq_update = torch.bincount(labels.view(-1), minlength=self.num_labels).float()
            
        self.label_frequencies += freq_update
        self.update_count += labels.size(0)
        
    def get_tail_mask(self) -> torch.Tensor:
        """Get boolean mask indicating which labels are considered tail labels."""
        if self.update_count == 0:
            return torch.zeros(self.num_labels, dtype=torch.bool, device=self.label_frequencies.device)
        
        normalized_freq = self.label_frequencies / self.update_count
        return normalized_freq < self.tail_threshold
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the ExectModel.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            token_type_ids: Token type IDs
            labels: Ground truth labels (optional, for training)
            return_dict: Whether to return a dictionary
            
        Returns:
            Dictionary containing logits, loss, and other outputs
        """
        # Get BERT outputs
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        
        sequence_output = bert_outputs.last_hidden_state
        pooled_output = bert_outputs.pooler_output
        
        # Multi-scale feature extraction
        features = []
        for extractor in self.feature_extractors:
            features.append(extractor(pooled_output))
            
        # Feature fusion
        concatenated_features = torch.cat(features, dim=-1)
        fused_features = self.feature_fusion(concatenated_features)
        
        # Apply tail-adaptive attention if enabled
        if self.use_tail_attention:
            tail_mask = self.get_tail_mask()
            attention_features = self.tail_attention(
                sequence_output, 
                pooled_output,
                tail_mask=tail_mask
            )
            # Combine original features with attention features
            final_features = torch.cat([fused_features, attention_features], dim=-1)
        else:
            final_features = fused_features
            
        final_features = self.dropout(final_features)
        
        # Multi-head classification
        head_logits = self.head_classifier(final_features)
        tail_logits = self.tail_classifier(final_features)
        
        # Combine head and tail logits based on label frequency
        if self.training and labels is not None:
            self.update_label_frequencies(labels)
            
        tail_mask = self.get_tail_mask()
        logits = torch.where(
            tail_mask.unsqueeze(0).expand_as(head_logits),
            tail_logits,
            head_logits
        )
        
        # Apply temperature scaling
        logits = logits / self.temperature
        
        # Calibrate predictions for better top-k performance
        calibrated_logits = self.calibrator(logits)
        
        outputs = {
            'logits': calibrated_logits,
            'raw_logits': logits,
            'head_logits': head_logits,
            'tail_logits': tail_logits,
        }
        
        # Compute loss if labels are provided
        if labels is not None:
            # Convert labels to multi-hot if needed
            if labels.dim() == 1:
                multi_hot_labels = torch.zeros(
                    labels.size(0), self.num_labels, 
                    device=labels.device, dtype=torch.float
                )
                multi_hot_labels.scatter_(1, labels.unsqueeze(1), 1.0)
            else:
                multi_hot_labels = labels.float()
            
            # Compute focused loss
            focused_loss = self.focused_loss(calibrated_logits, multi_hot_labels)
            
            # Compute contrastive loss for better class separation
            contrastive_loss = self.contrastive_loss(
                final_features, multi_hot_labels
            )
            
            # Combine losses
            total_loss = focused_loss + 0.1 * contrastive_loss
            
            outputs.update({
                'loss': total_loss,
                'focused_loss': focused_loss,
                'contrastive_loss': contrastive_loss,
            })
            
        if not return_dict:
            return tuple(outputs.values())
            
        return outputs
    
    def predict_top_k(
        self, 
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        k: int = 5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict top-k labels for input.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            k: Number of top predictions to return
            
        Returns:
            Tuple of (top_k_scores, top_k_indices)
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            logits = outputs['logits']
            probabilities = torch.sigmoid(logits)
            
            top_k_scores, top_k_indices = torch.topk(probabilities, k, dim=-1)
            
        return top_k_scores, top_k_indices