"""
Test suite for Enhanced ExectModel and related components.

This module contains comprehensive tests for all components of the
extreme multi-label classification system.
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from extreme_classification import (
    ExectModel,
    TailAdaptiveAttention,
    FocusedLoss,
    ContrastiveLoss,
    TopKCalibrator,
    PrecisionAtK,
    NDCGAtK,
    ExtremeMultiLabelMetrics
)


class TestExectModel(unittest.TestCase):
    """Test the main ExectModel implementation."""
    
    def setUp(self):
        self.batch_size = 4
        self.seq_length = 32
        self.num_labels = 100
        self.hidden_size = 768
        
        # Create mock config
        class MockConfig:
            hidden_size = 768
            
        self.config = MockConfig()
        
        # Initialize model with minimal configuration for testing
        self.model = ExectModel(
            config=self.config,
            num_labels=self.num_labels,
            bert_model_name=None,  # Don't load pretrained for tests
            use_tail_attention=True,
            dropout_rate=0.1
        )
        
        # Create sample inputs
        self.input_ids = torch.randint(0, 1000, (self.batch_size, self.seq_length))
        self.attention_mask = torch.ones(self.batch_size, self.seq_length)
        self.labels = torch.zeros(self.batch_size, self.num_labels)
        # Set some random positive labels
        for i in range(self.batch_size):
            num_pos = torch.randint(1, 5, (1,)).item()
            pos_indices = torch.randperm(self.num_labels)[:num_pos]
            self.labels[i, pos_indices] = 1.0
    
    def test_model_initialization(self):
        """Test model initialization."""
        self.assertEqual(self.model.num_labels, self.num_labels)
        self.assertEqual(self.model.hidden_size, self.hidden_size)
        self.assertTrue(self.model.use_tail_attention)
    
    def test_forward_pass_without_labels(self):
        """Test forward pass without labels."""
        with torch.no_grad():
            outputs = self.model(
                input_ids=self.input_ids,
                attention_mask=self.attention_mask
            )
        
        self.assertIn('logits', outputs)
        self.assertEqual(outputs['logits'].shape, (self.batch_size, self.num_labels))
        self.assertNotIn('loss', outputs)
    
    def test_forward_pass_with_labels(self):
        """Test forward pass with labels."""
        outputs = self.model(
            input_ids=self.input_ids,
            attention_mask=self.attention_mask,
            labels=self.labels
        )
        
        self.assertIn('logits', outputs)
        self.assertIn('loss', outputs)
        self.assertEqual(outputs['logits'].shape, (self.batch_size, self.num_labels))
        self.assertIsInstance(outputs['loss'], torch.Tensor)
    
    def test_predict_top_k(self):
        """Test top-k prediction functionality."""
        k = 5
        scores, indices = self.model.predict_top_k(
            input_ids=self.input_ids,
            attention_mask=self.attention_mask,
            k=k
        )
        
        self.assertEqual(scores.shape, (self.batch_size, k))
        self.assertEqual(indices.shape, (self.batch_size, k))
        
        # Check that scores are in descending order
        for i in range(self.batch_size):
            sorted_scores = torch.sort(scores[i], descending=True)[0]
            self.assertTrue(torch.allclose(scores[i], sorted_scores))
    
    def test_label_frequency_update(self):
        """Test label frequency tracking."""
        initial_freq = self.model.label_frequencies.clone()
        
        # Update frequencies
        self.model.update_label_frequencies(self.labels)
        
        # Check that frequencies were updated
        self.assertFalse(torch.equal(initial_freq, self.model.label_frequencies))
        self.assertGreater(self.model.update_count.item(), 0)
    
    def test_tail_mask_computation(self):
        """Test tail label mask computation."""
        # Update frequencies first
        self.model.update_label_frequencies(self.labels)
        
        tail_mask = self.model.get_tail_mask()
        self.assertEqual(tail_mask.shape, (self.num_labels,))
        self.assertEqual(tail_mask.dtype, torch.bool)


class TestTailAdaptiveAttention(unittest.TestCase):
    """Test TailAdaptiveAttention module."""
    
    def setUp(self):
        self.batch_size = 4
        self.seq_length = 32
        self.hidden_size = 768
        self.num_labels = 100
        
        self.attention = TailAdaptiveAttention(
            hidden_size=self.hidden_size,
            num_labels=self.num_labels,
            tail_threshold=0.01
        )
        
        self.sequence_output = torch.randn(self.batch_size, self.seq_length, self.hidden_size)
        self.pooled_output = torch.randn(self.batch_size, self.hidden_size)
        self.tail_mask = torch.randint(0, 2, (self.num_labels,)).bool()
    
    def test_attention_forward(self):
        """Test attention forward pass."""
        output = self.attention(
            sequence_output=self.sequence_output,
            pooled_output=self.pooled_output,
            tail_mask=self.tail_mask
        )
        
        self.assertEqual(output.shape, (self.batch_size, self.hidden_size))
    
    def test_tail_attention_weights(self):
        """Test tail attention weight computation."""
        weights = self.attention.compute_tail_attention_weights(
            self.sequence_output, self.tail_mask
        )
        
        self.assertEqual(weights.shape, (self.batch_size, self.seq_length))
        
        # Check that weights sum to 1 for each sample
        weight_sums = weights.sum(dim=-1)
        expected_sums = torch.ones(self.batch_size)
        self.assertTrue(torch.allclose(weight_sums, expected_sums, atol=1e-6))


class TestLossFunctions(unittest.TestCase):
    """Test loss functions."""
    
    def setUp(self):
        self.batch_size = 4
        self.num_labels = 100
        self.hidden_size = 768
        
        self.logits = torch.randn(self.batch_size, self.num_labels)
        self.targets = torch.zeros(self.batch_size, self.num_labels)
        # Set some random positive labels
        for i in range(self.batch_size):
            num_pos = torch.randint(1, 5, (1,)).item()
            pos_indices = torch.randperm(self.num_labels)[:num_pos]
            self.targets[i, pos_indices] = 1.0
            
        self.features = torch.randn(self.batch_size, self.hidden_size)
    
    def test_focused_loss(self):
        """Test FocusedLoss computation."""
        loss_fn = FocusedLoss(alpha=1.0, gamma=2.0)
        loss = loss_fn(self.logits, self.targets)
        
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0)  # Scalar loss
        self.assertGreater(loss.item(), 0)
    
    def test_contrastive_loss(self):
        """Test ContrastiveLoss computation."""
        loss_fn = ContrastiveLoss(temperature=0.1)
        loss = loss_fn(self.features, self.targets)
        
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0)  # Scalar loss
        self.assertGreaterEqual(loss.item(), 0)
    
    def test_focused_loss_with_different_parameters(self):
        """Test FocusedLoss with different parameters."""
        # Test different alpha values
        loss_fn1 = FocusedLoss(alpha=0.25, gamma=2.0)
        loss_fn2 = FocusedLoss(alpha=0.75, gamma=2.0)
        
        loss1 = loss_fn1(self.logits, self.targets)
        loss2 = loss_fn2(self.logits, self.targets)
        
        self.assertNotEqual(loss1.item(), loss2.item())
        
        # Test different gamma values
        loss_fn3 = FocusedLoss(alpha=1.0, gamma=1.0)
        loss_fn4 = FocusedLoss(alpha=1.0, gamma=3.0)
        
        loss3 = loss_fn3(self.logits, self.targets)
        loss4 = loss_fn4(self.logits, self.targets)
        
        self.assertNotEqual(loss3.item(), loss4.item())


class TestCalibration(unittest.TestCase):
    """Test calibration techniques."""
    
    def setUp(self):
        self.batch_size = 4
        self.num_labels = 100
        self.k_values = [1, 3, 5]
        
        self.calibrator = TopKCalibrator(
            num_labels=self.num_labels,
            k_values=self.k_values
        )
        
        self.logits = torch.randn(self.batch_size, self.num_labels)
    
    def test_temperature_scaling(self):
        """Test temperature scaling."""
        scaled_logits = self.calibrator.temperature_scale(self.logits)
        
        self.assertEqual(scaled_logits.shape, self.logits.shape)
        self.assertFalse(torch.equal(scaled_logits, self.logits))
    
    def test_calibrator_forward(self):
        """Test calibrator forward pass."""
        calibrated_logits = self.calibrator(self.logits)
        
        self.assertEqual(calibrated_logits.shape, self.logits.shape)
    
    def test_position_aware_calibration(self):
        """Test position-aware calibration."""
        k = 5
        calibrated_logits = self.calibrator.position_aware_calibration(self.logits, k)
        
        self.assertEqual(calibrated_logits.shape, self.logits.shape)
    
    def test_confidence_calibration(self):
        """Test confidence-based calibration."""
        probs = torch.sigmoid(self.logits)
        calibrated_probs = self.calibrator.confidence_calibration(probs)
        
        self.assertEqual(calibrated_probs.shape, probs.shape)


class TestMetrics(unittest.TestCase):
    """Test evaluation metrics."""
    
    def setUp(self):
        self.batch_size = 4
        self.num_labels = 20  # Smaller for easier testing
        self.k_values = [1, 3, 5]
        
        # Create deterministic test data
        torch.manual_seed(42)
        self.predictions = torch.randn(self.batch_size, self.num_labels)
        self.targets = torch.zeros(self.batch_size, self.num_labels)
        
        # Manually set some positive labels for predictable testing
        self.targets[0, [0, 2, 5]] = 1.0  # Sample 0 has labels 0, 2, 5
        self.targets[1, [1, 3]] = 1.0     # Sample 1 has labels 1, 3
        self.targets[2, [4, 6, 7, 8]] = 1.0  # Sample 2 has labels 4, 6, 7, 8
        self.targets[3, [9]] = 1.0        # Sample 3 has label 9
    
    def test_precision_at_k(self):
        """Test Precision@K computation."""
        metric = PrecisionAtK(k_values=self.k_values)
        metric.update(self.predictions, self.targets)
        results = metric.compute()
        
        for k in self.k_values:
            self.assertIn(f'precision_at_{k}', results)
            self.assertGreaterEqual(results[f'precision_at_{k}'], 0.0)
            self.assertLessEqual(results[f'precision_at_{k}'], 1.0)
    
    def test_ndcg_at_k(self):
        """Test NDCG@K computation."""
        metric = NDCGAtK(k_values=self.k_values)
        metric.update(self.predictions, self.targets)
        results = metric.compute()
        
        for k in self.k_values:
            self.assertIn(f'ndcg_at_{k}', results)
            self.assertGreaterEqual(results[f'ndcg_at_{k}'], 0.0)
            self.assertLessEqual(results[f'ndcg_at_{k}'], 1.0)
    
    def test_extreme_multilabel_metrics(self):
        """Test comprehensive metrics suite."""
        metrics = ExtremeMultiLabelMetrics(k_values=self.k_values)
        metrics.update(self.predictions, self.targets)
        
        results = metrics.compute()
        summary = metrics.compute_summary()
        
        # Check that results contain expected metrics
        expected_metrics = []
        for k in self.k_values:
            expected_metrics.extend([
                f'precision_at_{k}',
                f'ndcg_at_{k}',
                f'recall_at_{k}',
                f'f1_at_{k}'
            ])
        expected_metrics.append('mean_average_precision')
        
        for metric_name in expected_metrics:
            self.assertIn(metric_name, results)
        
        # Check summary metrics
        self.assertIn('precision_at_3', summary)
        self.assertIn('ndcg_at_5', summary)
    
    def test_metric_reset(self):
        """Test metric reset functionality."""
        metric = PrecisionAtK(k_values=[1, 3])
        
        # Update with some data
        metric.update(self.predictions, self.targets)
        self.assertGreater(metric.total_samples, 0)
        
        # Reset and check
        metric.reset()
        self.assertEqual(metric.total_samples, 0)
        for k in metric.k_values:
            self.assertEqual(metric.precision_sums[k], 0.0)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system."""
    
    def setUp(self):
        self.batch_size = 2
        self.seq_length = 16
        self.num_labels = 50
        self.hidden_size = 768
        
        # Create minimal mock BERT config
        class MockConfig:
            hidden_size = 768
        
        self.model = ExectModel(
            config=MockConfig(),
            num_labels=self.num_labels,
            bert_model_name=None,
            use_tail_attention=True
        )
        
        self.input_ids = torch.randint(0, 1000, (self.batch_size, self.seq_length))
        self.attention_mask = torch.ones(self.batch_size, self.seq_length)
        self.labels = torch.zeros(self.batch_size, self.num_labels)
        self.labels[0, [0, 2, 5]] = 1.0
        self.labels[1, [1, 3, 7]] = 1.0
    
    def test_end_to_end_training_step(self):
        """Test complete training step."""
        self.model.train()
        
        # Forward pass
        outputs = self.model(
            input_ids=self.input_ids,
            attention_mask=self.attention_mask,
            labels=self.labels
        )
        
        loss = outputs['loss']
        
        # Backward pass
        loss.backward()
        
        # Check that gradients were computed
        for param in self.model.parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)
    
    def test_end_to_end_evaluation(self):
        """Test complete evaluation pipeline."""
        self.model.eval()
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=self.input_ids,
                attention_mask=self.attention_mask
            )
        
        logits = outputs['logits']
        
        # Evaluate with metrics
        metrics = ExtremeMultiLabelMetrics(k_values=[1, 3, 5])
        metrics.update(logits, self.labels)
        results = metrics.compute()
        
        # Check that we got valid results
        self.assertIn('precision_at_3', results)
        self.assertIn('ndcg_at_5', results)
        self.assertGreaterEqual(results['precision_at_3'], 0.0)
        self.assertGreaterEqual(results['ndcg_at_5'], 0.0)


def run_tests():
    """Run all tests."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestExectModel,
        TestTailAdaptiveAttention,
        TestLossFunctions,
        TestCalibration,
        TestMetrics,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    if success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")
        exit(1)