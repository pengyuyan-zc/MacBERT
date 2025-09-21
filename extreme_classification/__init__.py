"""
Extreme Multi-Label Classification with Enhanced ExectModel
=========================================================

This module provides an improved implementation of ExectModel for extreme multi-label 
classification tasks, focusing on enhancing Precision@3 and NDCG@5 metrics.

Key Features:
- Enhanced architecture with better tail label handling
- TailAdaptiveAttention for rare class detection
- Focused loss function for challenging examples
- Contrastive learning for better class separation
- Calibration techniques for top-k prediction accuracy
"""

from .model import ExectModel
from .attention import TailAdaptiveAttention
from .loss import FocusedLoss, ContrastiveLoss
from .calibration import TopKCalibrator
from .metrics import PrecisionAtK, NDCGAtK, ExtremeMultiLabelMetrics

__version__ = "1.0.0"
__all__ = [
    "ExectModel",
    "TailAdaptiveAttention", 
    "FocusedLoss",
    "ContrastiveLoss",
    "TopKCalibrator",
    "PrecisionAtK",
    "NDCGAtK",
    "ExtremeMultiLabelMetrics"
]