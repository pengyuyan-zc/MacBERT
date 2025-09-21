"""
Specialized Loss Functions for Extreme Multi-Label Classification

This module implements focused loss functions that put more emphasis on 
challenging examples and contrastive learning for better class separation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import math


class FocusedLoss(nn.Module):
    """
    Focused Loss for extreme multi-label classification.
    
    This loss function puts more emphasis on challenging examples by down-weighting
    easy examples and focusing learning on hard negatives and positives.
    
    Args:
        alpha: Weighting factor for positive/negative balance
        gamma: Focusing parameter (higher gamma = more focus on hard examples)
        reduction: Loss reduction method
        label_smoothing: Label smoothing factor
    """
    
    def __init__(
        self,
        alpha: float = 1.0,
        gamma: float = 2.0,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
        pos_weight: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.register_buffer('pos_weight', pos_weight)
        
    def forward(
        self, 
        logits: torch.Tensor, 
        targets: torch.Tensor,
        class_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute focused loss.
        
        Args:
            logits: Model predictions [batch_size, num_labels]
            targets: Ground truth labels [batch_size, num_labels]
            class_weights: Per-class weights [num_labels]
            
        Returns:
            Computed loss value
        """
        # Apply label smoothing if specified
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + \
                     self.label_smoothing / targets.size(-1)
        
        # Compute probabilities
        probs = torch.sigmoid(logits)
        
        # Compute binary cross entropy
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction='none'
        )
        
        # Compute focal weights
        # For positive examples: (1 - p)^gamma
        # For negative examples: p^gamma
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weights = (1 - pt) ** self.gamma
        
        # Apply alpha weighting
        if self.alpha != 1.0:
            alpha_weights = torch.where(targets == 1, self.alpha, 1 - self.alpha)
            focal_weights = alpha_weights * focal_weights
        
        # Apply class weights if provided
        if class_weights is not None:
            class_weights = class_weights.unsqueeze(0).expand_as(targets)
            focal_weights = focal_weights * class_weights
        
        # Compute focused loss
        focused_loss = focal_weights * bce_loss
        
        if self.reduction == "mean":
            return focused_loss.mean()
        elif self.reduction == "sum":
            return focused_loss.sum()
        else:
            return focused_loss


class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss for better class separation.
    
    This loss encourages the model to learn representations where similar
    classes are closer together and dissimilar classes are further apart.
    
    Args:
        temperature: Temperature parameter for softmax
        margin: Margin for contrastive learning
        use_cosine_similarity: Whether to use cosine similarity
    """
    
    def __init__(
        self,
        temperature: float = 0.1,
        margin: float = 0.5,
        use_cosine_similarity: bool = True
    ):
        super().__init__()
        self.temperature = temperature
        self.margin = margin
        self.use_cosine_similarity = use_cosine_similarity
        
    def compute_similarity(
        self, 
        features: torch.Tensor, 
        normalize: bool = True
    ) -> torch.Tensor:
        """Compute pairwise similarity matrix."""
        if normalize and self.use_cosine_similarity:
            features = F.normalize(features, p=2, dim=-1)
            similarity = torch.matmul(features, features.t())
        else:
            similarity = torch.matmul(features, features.t())
            
        return similarity
    
    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        feature_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute contrastive loss.
        
        Args:
            features: Feature representations [batch_size, hidden_size]
            labels: Ground truth labels [batch_size, num_labels]
            feature_weights: Weights for features [batch_size]
            
        Returns:
            Contrastive loss value
        """
        batch_size = features.size(0)
        
        # Compute similarity matrix
        similarity = self.compute_similarity(features) / self.temperature
        
        # Create mask for positive pairs (samples with overlapping labels)
        label_similarity = torch.matmul(labels, labels.t())  # [batch_size, batch_size]
        positive_mask = (label_similarity > 0).float()
        
        # Remove self-similarity
        eye = torch.eye(batch_size, device=features.device)
        positive_mask = positive_mask * (1 - eye)
        negative_mask = (1 - positive_mask) * (1 - eye)
        
        # Compute positive and negative similarities
        positive_similarities = similarity * positive_mask
        negative_similarities = similarity * negative_mask
        
        # InfoNCE-style contrastive loss
        # For each sample, compute loss against all other samples
        losses = []
        
        for i in range(batch_size):
            # Get positive and negative similarities for sample i
            pos_sims = positive_similarities[i][positive_mask[i] > 0]
            neg_sims = negative_similarities[i][negative_mask[i] > 0]
            
            if len(pos_sims) > 0:
                # Compute InfoNCE loss
                all_sims = torch.cat([pos_sims, neg_sims])
                pos_exp = torch.exp(pos_sims).sum()
                all_exp = torch.exp(all_sims).sum()
                
                loss = -torch.log(pos_exp / (all_exp + 1e-8))
                
                if feature_weights is not None:
                    loss = loss * feature_weights[i]
                    
                losses.append(loss)
        
        if len(losses) > 0:
            return torch.stack(losses).mean()
        else:
            return torch.tensor(0.0, device=features.device, requires_grad=True)


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for imbalanced multi-label classification.
    
    This loss addresses the imbalance between positive and negative samples
    by applying different focusing strategies to positive and negative examples.
    """
    
    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        clip: float = 0.05,
        eps: float = 1e-8,
        disable_torch_grad_focal_loss: bool = False
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute asymmetric loss.
        
        Args:
            logits: Model predictions [batch_size, num_labels]
            targets: Ground truth labels [batch_size, num_labels]
            
        Returns:
            Asymmetric loss value
        """
        # Probabilities and complementary probabilities
        xs_pos = torch.sigmoid(logits)
        xs_neg = 1 - xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Calculate Losses
        los_pos = targets * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - targets) * torch.log(xs_neg.clamp(min=self.eps))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * targets
            pt1 = xs_neg * (1 - targets)
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * targets + self.gamma_neg * (1 - targets)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss = one_sided_w * (los_pos + los_neg)
        else:
            loss = los_pos + los_neg

        return -loss.mean()


class RankingLoss(nn.Module):
    """
    Ranking Loss for improving top-k performance.
    
    This loss encourages the model to rank relevant labels higher than
    irrelevant ones, which is crucial for metrics like Precision@k and NDCG@k.
    """
    
    def __init__(
        self,
        margin: float = 1.0,
        k: int = 5,
        temperature: float = 1.0
    ):
        super().__init__()
        self.margin = margin
        self.k = k
        self.temperature = temperature
        
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        sample_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute ranking loss.
        
        Args:
            logits: Model predictions [batch_size, num_labels]
            targets: Ground truth labels [batch_size, num_labels]
            sample_weights: Per-sample weights [batch_size]
            
        Returns:
            Ranking loss value
        """
        batch_size, num_labels = logits.shape
        
        # Apply temperature scaling
        scaled_logits = logits / self.temperature
        
        # Separate positive and negative scores
        positive_mask = targets > 0
        negative_mask = targets == 0
        
        losses = []
        
        for i in range(batch_size):
            pos_scores = scaled_logits[i][positive_mask[i]]
            neg_scores = scaled_logits[i][negative_mask[i]]
            
            if len(pos_scores) > 0 and len(neg_scores) > 0:
                # Compute pairwise ranking loss
                pos_expanded = pos_scores.unsqueeze(1)  # [num_pos, 1]
                neg_expanded = neg_scores.unsqueeze(0)  # [1, num_neg]
                
                # Hinge loss: max(0, margin - (pos_score - neg_score))
                pairwise_loss = F.relu(
                    self.margin - (pos_expanded - neg_expanded)
                )
                
                sample_loss = pairwise_loss.mean()
                
                if sample_weights is not None:
                    sample_loss = sample_loss * sample_weights[i]
                    
                losses.append(sample_loss)
        
        if len(losses) > 0:
            return torch.stack(losses).mean()
        else:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)


class CombinedLoss(nn.Module):
    """
    Combined loss function that integrates multiple loss components
    for comprehensive extreme multi-label classification training.
    """
    
    def __init__(
        self,
        focused_weight: float = 1.0,
        contrastive_weight: float = 0.1,
        asymmetric_weight: float = 0.5,
        ranking_weight: float = 0.2,
        **kwargs
    ):
        super().__init__()
        self.focused_weight = focused_weight
        self.contrastive_weight = contrastive_weight
        self.asymmetric_weight = asymmetric_weight
        self.ranking_weight = ranking_weight
        
        self.focused_loss = FocusedLoss(**kwargs.get('focused_kwargs', {}))
        self.contrastive_loss = ContrastiveLoss(**kwargs.get('contrastive_kwargs', {}))
        self.asymmetric_loss = AsymmetricLoss(**kwargs.get('asymmetric_kwargs', {}))
        self.ranking_loss = RankingLoss(**kwargs.get('ranking_kwargs', {}))
        
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        features: Optional[torch.Tensor] = None,
        return_components: bool = False
    ) -> torch.Tensor:
        """
        Compute combined loss.
        
        Args:
            logits: Model predictions [batch_size, num_labels]
            targets: Ground truth labels [batch_size, num_labels]
            features: Feature representations [batch_size, hidden_size]
            return_components: Whether to return individual loss components
            
        Returns:
            Combined loss (and components if requested)
        """
        components = {}
        
        # Focused loss
        components['focused'] = self.focused_loss(logits, targets)
        
        # Asymmetric loss
        components['asymmetric'] = self.asymmetric_loss(logits, targets)
        
        # Ranking loss
        components['ranking'] = self.ranking_loss(logits, targets)
        
        # Contrastive loss (requires features)
        if features is not None:
            components['contrastive'] = self.contrastive_loss(features, targets)
        else:
            components['contrastive'] = torch.tensor(0.0, device=logits.device)
        
        # Combine losses
        total_loss = (
            self.focused_weight * components['focused'] +
            self.asymmetric_weight * components['asymmetric'] +
            self.ranking_weight * components['ranking'] +
            self.contrastive_weight * components['contrastive']
        )
        
        if return_components:
            components['total'] = total_loss
            return components
        else:
            return total_loss