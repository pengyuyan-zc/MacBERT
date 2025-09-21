"""
Basic validation script for the Enhanced ExectModel implementation.

This script performs basic validation of the implementation without 
requiring expensive model downloads or full dependency installation.
"""

import sys
import os
import torch
import numpy as np
from typing import Dict, List

# Add the project root to path
sys.path.append('/home/runner/work/MacBERT/MacBERT')

def test_imports():
    """Test that all modules can be imported correctly."""
    print("Testing imports...")
    
    try:
        # Test core modules
        from extreme_classification.model import ExectModel
        from extreme_classification.attention import TailAdaptiveAttention
        from extreme_classification.loss import FocusedLoss, ContrastiveLoss
        from extreme_classification.calibration import TopKCalibrator
        from extreme_classification.metrics import PrecisionAtK, NDCGAtK
        
        print("✅ All imports successful!")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality without requiring pretrained models."""
    print("\nTesting basic functionality...")
    
    try:
        from extreme_classification.metrics import PrecisionAtK, NDCGAtK
        from extreme_classification.loss import FocusedLoss
        from extreme_classification.calibration import TopKCalibrator
        
        # Test metrics
        batch_size, num_labels = 4, 20
        predictions = torch.randn(batch_size, num_labels)
        targets = torch.zeros(batch_size, num_labels)
        targets[0, [0, 2, 5]] = 1.0
        targets[1, [1, 3]] = 1.0
        
        # Test Precision@K
        precision_metric = PrecisionAtK(k_values=[1, 3, 5])
        precision_metric.update(predictions, targets)
        precision_results = precision_metric.compute()
        
        assert 'precision_at_3' in precision_results
        assert 0 <= precision_results['precision_at_3'] <= 1
        
        # Test NDCG@K
        ndcg_metric = NDCGAtK(k_values=[1, 3, 5])
        ndcg_metric.update(predictions, targets)
        ndcg_results = ndcg_metric.compute()
        
        assert 'ndcg_at_5' in ndcg_results
        assert 0 <= ndcg_results['ndcg_at_5'] <= 1
        
        # Test loss functions
        loss_fn = FocusedLoss(alpha=1.0, gamma=2.0)
        loss = loss_fn(predictions, targets)
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0
        
        # Test calibration
        calibrator = TopKCalibrator(num_labels=num_labels, k_values=[1, 3, 5])
        calibrated = calibrator(predictions)
        assert calibrated.shape == predictions.shape
        
        print("✅ Basic functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Functionality test error: {e}")
        return False

def test_model_architecture():
    """Test model architecture without pretrained weights."""
    print("\nTesting model architecture...")
    
    try:
        from extreme_classification.model import ExectModel
        from extreme_classification.attention import TailAdaptiveAttention
        from transformers import BertConfig
        
        # Test TailAdaptiveAttention
        attention = TailAdaptiveAttention(
            hidden_size=768,
            num_labels=100,
            tail_threshold=0.01
        )
        
        batch_size, seq_len, hidden_size = 2, 16, 768
        sequence_output = torch.randn(batch_size, seq_len, hidden_size)
        pooled_output = torch.randn(batch_size, hidden_size)
        tail_mask = torch.randint(0, 2, (100,)).bool()
        
        attention_output = attention(
            sequence_output=sequence_output,
            pooled_output=pooled_output,
            tail_mask=tail_mask
        )
        
        assert attention_output.shape == (batch_size, hidden_size)
        
        # Test ExectModel architecture components
        config = BertConfig(hidden_size=768)
        model = ExectModel(
            config=config,
            num_labels=100,
            bert_model_name=None,  # Don't load pretrained
            use_tail_attention=True
        )
        
        # Test model components exist
        assert hasattr(model, 'feature_extractors')
        assert hasattr(model, 'tail_attention')
        assert hasattr(model, 'head_classifier')
        assert hasattr(model, 'tail_classifier')
        assert hasattr(model, 'calibrator')
        
        print("✅ Model architecture tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Model architecture test error: {e}")
        return False

def test_metric_calculations():
    """Test specific metric calculation correctness."""
    print("\nTesting metric calculations...")
    
    try:
        from extreme_classification.metrics import PrecisionAtK, NDCGAtK
        
        # Create simple test case with known results
        batch_size, num_labels = 2, 10
        predictions = torch.tensor([
            [0.9, 0.8, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.0],  # Top-3: [0, 1, 8]
            [0.1, 0.9, 0.8, 0.7, 0.2, 0.3, 0.4, 0.5, 0.6, 0.0]   # Top-3: [1, 2, 3]
        ])
        targets = torch.tensor([
            [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Labels: [0, 1]
            [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]   # Labels: [1, 3]
        ])
        
        # Test Precision@3
        precision_metric = PrecisionAtK(k_values=[3])
        precision_metric.update(predictions, targets)
        results = precision_metric.compute()
        
        # For sample 0: top-3 are [0, 1, 8], relevant are [0, 1] -> precision = 2/3
        # For sample 1: top-3 are [1, 2, 3], relevant are [1, 3] -> precision = 2/3
        # Average: (2/3 + 2/3) / 2 = 2/3 ≈ 0.667
        expected_precision = 2.0 / 3.0
        actual_precision = results['precision_at_3']
        
        assert abs(actual_precision - expected_precision) < 1e-6, \
            f"Expected {expected_precision}, got {actual_precision}"
        
        print("✅ Metric calculation tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Metric calculation test error: {e}")
        return False

def validate_implementation():
    """Run all validation tests."""
    print("=== Enhanced ExectModel Validation ===\n")
    
    tests = [
        test_imports,
        test_basic_functionality,
        test_model_architecture,
        test_metric_calculations
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print(f"\n=== Validation Summary ===")
    print(f"Tests passed: {sum(results)}/{len(results)}")
    
    if all(results):
        print("🎉 All validation tests passed! Implementation is working correctly.")
        return True
    else:
        print("⚠️ Some validation tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = validate_implementation()
    sys.exit(0 if success else 1)