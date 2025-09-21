"""
Calibration Techniques for Top-k Prediction Accuracy

This module implements calibration methods specifically optimized for 
improving top-k prediction accuracy in extreme multi-label classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Dict
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import CalibratedClassifierCV


class TopKCalibrator(nn.Module):
    """
    Top-k Calibrator for improving prediction accuracy at top positions.
    
    This module implements various calibration techniques to improve the
    ranking quality and probability estimates for top-k predictions.
    
    Args:
        num_labels: Number of labels
        k_values: List of k values to optimize for
        calibration_method: Method for calibration ('temperature', 'platt', 'isotonic')
        temperature_init: Initial temperature for temperature scaling
    """
    
    def __init__(
        self,
        num_labels: int,
        k_values: List[int] = [1, 3, 5],
        calibration_method: str = "temperature",
        temperature_init: float = 1.0,
        use_bias: bool = True
    ):
        super().__init__()
        self.num_labels = num_labels
        self.k_values = k_values
        self.calibration_method = calibration_method
        self.use_bias = use_bias
        
        # Temperature scaling parameters
        self.temperature = nn.Parameter(torch.ones(1) * temperature_init)
        
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(num_labels))
        
        # Position-dependent calibration
        self.position_weights = nn.Parameter(
            torch.ones(max(k_values))
        ) if k_values else nn.Parameter(torch.ones(1))
        
        # Label-specific calibration parameters
        self.label_temperature = nn.Parameter(
            torch.ones(num_labels) * temperature_init
        )
        
        # Confidence-based calibration
        self.confidence_calibrator = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
        # Ranking calibration network
        self.ranking_calibrator = nn.Sequential(
            nn.Linear(num_labels, num_labels // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(num_labels // 4, num_labels),
            nn.Tanh()
        )
        
    def temperature_scale(
        self, 
        logits: torch.Tensor, 
        use_label_specific: bool = True
    ) -> torch.Tensor:
        """Apply temperature scaling to logits."""
        if use_label_specific:
            # Use label-specific temperatures
            scaled_logits = logits / self.label_temperature.unsqueeze(0)
        else:
            # Use global temperature
            scaled_logits = logits / self.temperature
            
        if self.use_bias:
            scaled_logits = scaled_logits + self.bias.unsqueeze(0)
            
        return scaled_logits
    
    def confidence_calibration(self, probs: torch.Tensor) -> torch.Tensor:
        """Apply confidence-based calibration."""
        # Calculate confidence as max probability
        max_probs = probs.max(dim=-1, keepdim=True)[0]
        
        # Apply confidence calibration
        confidence_adjustment = self.confidence_calibrator(max_probs)
        
        # Adjust probabilities based on confidence
        calibrated_probs = probs * confidence_adjustment
        
        return calibrated_probs
    
    def ranking_calibration(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply ranking-aware calibration."""
        batch_size = logits.size(0)
        
        # Apply ranking calibration to adjust relative scores
        ranking_adjustment = self.ranking_calibrator(logits)
        
        # Combine original logits with ranking adjustment
        calibrated_logits = logits + 0.1 * ranking_adjustment
        
        return calibrated_logits
    
    def position_aware_calibration(
        self, 
        logits: torch.Tensor, 
        k: int
    ) -> torch.Tensor:
        """Apply position-aware calibration for top-k predictions."""
        batch_size = logits.size(0)
        
        # Get top-k predictions
        top_k_values, top_k_indices = torch.topk(logits, k, dim=-1)
        
        # Apply position-dependent weights
        if k <= len(self.position_weights):
            position_weights = self.position_weights[:k].unsqueeze(0)
            calibrated_top_k = top_k_values * position_weights
            
            # Update the original logits with calibrated top-k values
            calibrated_logits = logits.clone()
            calibrated_logits.scatter_(1, top_k_indices, calibrated_top_k)
        else:
            calibrated_logits = logits
            
        return calibrated_logits
    
    def forward(
        self, 
        logits: torch.Tensor, 
        apply_all: bool = True
    ) -> torch.Tensor:
        """
        Apply calibration to logits.
        
        Args:
            logits: Raw logits [batch_size, num_labels]
            apply_all: Whether to apply all calibration methods
            
        Returns:
            Calibrated logits
        """
        calibrated_logits = logits
        
        if self.calibration_method == "temperature" or apply_all:
            calibrated_logits = self.temperature_scale(calibrated_logits)
        
        if apply_all:
            # Apply ranking calibration
            calibrated_logits = self.ranking_calibration(calibrated_logits)
            
            # Apply position-aware calibration for the largest k value
            if self.k_values:
                max_k = max(self.k_values)
                calibrated_logits = self.position_aware_calibration(
                    calibrated_logits, max_k
                )
        
        return calibrated_logits
    
    def calibrate_probabilities(self, probs: torch.Tensor) -> torch.Tensor:
        """Apply post-hoc probability calibration."""
        return self.confidence_calibration(probs)


class PlattScaling(nn.Module):
    """
    Platt Scaling for probability calibration.
    
    Fits a sigmoid function to the outputs to calibrate probabilities.
    """
    
    def __init__(self, num_labels: int):
        super().__init__()
        self.A = nn.Parameter(torch.ones(num_labels))
        self.B = nn.Parameter(torch.zeros(num_labels))
        
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply Platt scaling."""
        return torch.sigmoid(self.A.unsqueeze(0) * logits + self.B.unsqueeze(0))


class IsotonicCalibrator(nn.Module):
    """
    Isotonic Regression based calibrator.
    
    Uses isotonic regression to learn a monotonic mapping from scores to probabilities.
    """
    
    def __init__(self, num_bins: int = 10):
        super().__init__()
        self.num_bins = num_bins
        self.bin_boundaries = nn.Parameter(
            torch.linspace(0, 1, num_bins + 1)[1:-1]
        )
        self.bin_values = nn.Parameter(torch.linspace(0, 1, num_bins))
        
    def forward(self, probs: torch.Tensor) -> torch.Tensor:
        """Apply isotonic calibration."""
        # Find which bin each probability belongs to
        bin_indices = torch.searchsorted(self.bin_boundaries, probs)
        
        # Clamp to valid range
        bin_indices = torch.clamp(bin_indices, 0, self.num_bins - 1)
        
        # Get calibrated values
        calibrated_probs = self.bin_values[bin_indices]
        
        return calibrated_probs


class MultiScaleCalibrator(nn.Module):
    """
    Multi-scale calibrator that applies different calibration strategies
    based on prediction confidence and label frequency.
    """
    
    def __init__(
        self,
        num_labels: int,
        num_confidence_bins: int = 5,
        num_frequency_bins: int = 3
    ):
        super().__init__()
        self.num_labels = num_labels
        self.num_confidence_bins = num_confidence_bins
        self.num_frequency_bins = num_frequency_bins
        
        # Confidence-based calibrators
        self.confidence_calibrators = nn.ModuleList([
            TopKCalibrator(num_labels, k_values=[1, 3, 5])
            for _ in range(num_confidence_bins)
        ])
        
        # Frequency-based calibrators
        self.frequency_calibrators = nn.ModuleList([
            TopKCalibrator(num_labels, k_values=[1, 3, 5])
            for _ in range(num_frequency_bins)
        ])
        
        # Gating network to select appropriate calibrator
        self.confidence_gate = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, num_confidence_bins),
            nn.Softmax(dim=-1)
        )
        
        self.frequency_gate = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, num_frequency_bins),
            nn.Softmax(dim=-1)
        )
        
    def forward(
        self,
        logits: torch.Tensor,
        label_frequencies: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply multi-scale calibration.
        
        Args:
            logits: Raw logits [batch_size, num_labels]
            label_frequencies: Label frequency information [num_labels]
            
        Returns:
            Calibrated logits
        """
        batch_size = logits.size(0)
        probs = torch.sigmoid(logits)
        
        # Confidence-based calibration
        max_confidence = probs.max(dim=-1, keepdim=True)[0]  # [batch_size, 1]
        confidence_weights = self.confidence_gate(max_confidence)  # [batch_size, num_confidence_bins]
        
        confidence_calibrated = torch.zeros_like(logits)
        for i, calibrator in enumerate(self.confidence_calibrators):
            calibrated = calibrator(logits)
            confidence_calibrated += confidence_weights[:, i:i+1] * calibrated
        
        # Frequency-based calibration if available
        if label_frequencies is not None:
            avg_frequency = label_frequencies.mean().unsqueeze(0).unsqueeze(0)  # [1, 1]
            frequency_weights = self.frequency_gate(avg_frequency)  # [1, num_frequency_bins]
            frequency_weights = frequency_weights.expand(batch_size, -1)
            
            frequency_calibrated = torch.zeros_like(logits)
            for i, calibrator in enumerate(self.frequency_calibrators):
                calibrated = calibrator(logits)
                frequency_calibrated += frequency_weights[:, i:i+1] * calibrated
            
            # Combine confidence and frequency calibration
            final_calibrated = 0.7 * confidence_calibrated + 0.3 * frequency_calibrated
        else:
            final_calibrated = confidence_calibrated
        
        return final_calibrated


class AdaptiveThresholdCalibrator(nn.Module):
    """
    Adaptive threshold calibrator that learns optimal thresholds for
    different k values and label characteristics.
    """
    
    def __init__(
        self,
        num_labels: int,
        k_values: List[int] = [1, 3, 5],
        num_threshold_bins: int = 10
    ):
        super().__init__()
        self.num_labels = num_labels
        self.k_values = k_values
        self.num_threshold_bins = num_threshold_bins
        
        # Learnable thresholds for different k values
        self.thresholds = nn.ParameterDict({
            f'k_{k}': nn.Parameter(torch.ones(num_labels) * 0.5)
            for k in k_values
        })
        
        # Threshold adjustment network
        self.threshold_adjuster = nn.Sequential(
            nn.Linear(num_labels + 1, num_labels // 4),  # +1 for k value
            nn.ReLU(),
            nn.Linear(num_labels // 4, num_labels),
            nn.Sigmoid()
        )
        
    def get_adaptive_thresholds(
        self,
        logits: torch.Tensor,
        k: int
    ) -> torch.Tensor:
        """Get adaptive thresholds for given k value."""
        batch_size = logits.size(0)
        
        # Base thresholds for this k
        base_thresholds = self.thresholds[f'k_{k}']
        
        # Prepare input for threshold adjuster
        k_tensor = torch.full((batch_size, 1), k / max(self.k_values), device=logits.device)
        adjuster_input = torch.cat([logits, k_tensor], dim=-1)
        
        # Get threshold adjustments
        threshold_adjustments = self.threshold_adjuster(adjuster_input)
        
        # Apply adjustments to base thresholds
        adaptive_thresholds = base_thresholds.unsqueeze(0) * threshold_adjustments
        
        return adaptive_thresholds
    
    def forward(
        self,
        logits: torch.Tensor,
        k: Optional[int] = None
    ) -> torch.Tensor:
        """
        Apply adaptive threshold calibration.
        
        Args:
            logits: Raw logits [batch_size, num_labels]
            k: Target k value for top-k prediction
            
        Returns:
            Threshold-calibrated logits
        """
        if k is None:
            k = max(self.k_values)
        
        if f'k_{k}' not in self.thresholds:
            k = min(self.k_values, key=lambda x: abs(x - k))
        
        # Get adaptive thresholds
        adaptive_thresholds = self.get_adaptive_thresholds(logits, k)
        
        # Apply threshold-based calibration
        probs = torch.sigmoid(logits)
        calibrated_probs = torch.where(
            probs > adaptive_thresholds,
            probs,
            probs * 0.5  # Reduce confidence for below-threshold predictions
        )
        
        # Convert back to logits
        calibrated_logits = torch.logit(
            torch.clamp(calibrated_probs, 1e-7, 1 - 1e-7)
        )
        
        return calibrated_logits