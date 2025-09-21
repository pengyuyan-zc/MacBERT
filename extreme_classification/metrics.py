"""
Evaluation Metrics for Extreme Multi-Label Classification

This module implements specialized metrics for evaluating extreme multi-label
classification performance, with focus on Precision@k and NDCG@k metrics.
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Union, Tuple
from sklearn.metrics import ndcg_score, precision_score, recall_score, f1_score
import warnings


class PrecisionAtK:
    """
    Precision@K metric for multi-label classification.
    
    Computes the precision of the top-k predictions, which is the fraction
    of relevant items in the top-k predictions.
    """
    
    def __init__(self, k_values: List[int] = [1, 3, 5, 10]):
        self.k_values = k_values
        self.reset()
        
    def reset(self):
        """Reset metric state."""
        self.total_samples = 0
        self.precision_sums = {k: 0.0 for k in self.k_values}
        
    def update(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor
    ):
        """
        Update metric with batch predictions and targets.
        
        Args:
            predictions: Model predictions [batch_size, num_labels]
            targets: Ground truth labels [batch_size, num_labels]
        """
        batch_size = predictions.size(0)
        self.total_samples += batch_size
        
        # Convert to numpy for easier processing
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
            
        for k in self.k_values:
            # Get top-k predictions for each sample
            top_k_indices = np.argsort(predictions, axis=1)[:, -k:]
            
            for i in range(batch_size):
                # Get relevant labels for this sample
                relevant_labels = np.where(targets[i] > 0)[0]
                predicted_labels = top_k_indices[i]
                
                if len(relevant_labels) > 0:
                    # Calculate precision@k
                    intersection = np.intersect1d(relevant_labels, predicted_labels)
                    precision_k = len(intersection) / k
                    self.precision_sums[k] += precision_k
                    
    def compute(self) -> Dict[str, float]:
        """Compute final precision@k values."""
        if self.total_samples == 0:
            return {f'precision_at_{k}': 0.0 for k in self.k_values}
            
        results = {}
        for k in self.k_values:
            results[f'precision_at_{k}'] = self.precision_sums[k] / self.total_samples
            
        return results


class NDCGAtK:
    """
    Normalized Discounted Cumulative Gain at K metric.
    
    Computes NDCG@K which measures the quality of ranking by considering
    both relevance and position of predictions.
    """
    
    def __init__(self, k_values: List[int] = [1, 3, 5, 10]):
        self.k_values = k_values
        self.reset()
        
    def reset(self):
        """Reset metric state."""
        self.total_samples = 0
        self.ndcg_sums = {k: 0.0 for k in self.k_values}
        
    def _dcg_at_k(self, relevances: np.ndarray, k: int) -> float:
        """Compute DCG@K."""
        relevances = relevances[:k]
        if relevances.size == 0:
            return 0.0
            
        # DCG formula: sum(rel_i / log2(i + 2)) for i in [0, k-1]
        discounts = np.log2(np.arange(2, k + 2))
        dcg = np.sum(relevances / discounts)
        return dcg
    
    def _ndcg_at_k(
        self, 
        predictions: np.ndarray, 
        targets: np.ndarray, 
        k: int
    ) -> float:
        """Compute NDCG@K for a single sample."""
        # Get top-k predictions
        top_k_indices = np.argsort(predictions)[-k:][::-1]  # Descending order
        
        # Get relevances for top-k predictions
        relevances = targets[top_k_indices]
        
        # Compute DCG@K
        dcg = self._dcg_at_k(relevances, k)
        
        # Compute IDCG@K (ideal DCG)
        ideal_relevances = np.sort(targets)[-k:][::-1]  # Best possible ranking
        idcg = self._dcg_at_k(ideal_relevances, k)
        
        if idcg == 0:
            return 0.0
            
        return dcg / idcg
    
    def update(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor
    ):
        """
        Update metric with batch predictions and targets.
        
        Args:
            predictions: Model predictions [batch_size, num_labels]  
            targets: Ground truth labels [batch_size, num_labels]
        """
        batch_size = predictions.size(0)
        self.total_samples += batch_size
        
        # Convert to numpy for easier processing
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
            
        for k in self.k_values:
            for i in range(batch_size):
                ndcg_k = self._ndcg_at_k(predictions[i], targets[i], k)
                self.ndcg_sums[k] += ndcg_k
                
    def compute(self) -> Dict[str, float]:
        """Compute final NDCG@k values."""
        if self.total_samples == 0:
            return {f'ndcg_at_{k}': 0.0 for k in self.k_values}
            
        results = {}
        for k in self.k_values:
            results[f'ndcg_at_{k}'] = self.ndcg_sums[k] / self.total_samples
            
        return results


class RecallAtK:
    """
    Recall@K metric for multi-label classification.
    
    Computes the recall of the top-k predictions, which is the fraction
    of relevant items that appear in the top-k predictions.
    """
    
    def __init__(self, k_values: List[int] = [1, 3, 5, 10]):
        self.k_values = k_values
        self.reset()
        
    def reset(self):
        """Reset metric state."""
        self.total_samples = 0
        self.recall_sums = {k: 0.0 for k in self.k_values}
        
    def update(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor
    ):
        """
        Update metric with batch predictions and targets.
        
        Args:
            predictions: Model predictions [batch_size, num_labels]
            targets: Ground truth labels [batch_size, num_labels]
        """
        batch_size = predictions.size(0)
        self.total_samples += batch_size
        
        # Convert to numpy for easier processing
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
            
        for k in self.k_values:
            # Get top-k predictions for each sample
            top_k_indices = np.argsort(predictions, axis=1)[:, -k:]
            
            for i in range(batch_size):
                # Get relevant labels for this sample
                relevant_labels = np.where(targets[i] > 0)[0]
                predicted_labels = top_k_indices[i]
                
                if len(relevant_labels) > 0:
                    # Calculate recall@k
                    intersection = np.intersect1d(relevant_labels, predicted_labels)
                    recall_k = len(intersection) / len(relevant_labels)
                    self.recall_sums[k] += recall_k
                    
    def compute(self) -> Dict[str, float]:
        """Compute final recall@k values."""
        if self.total_samples == 0:
            return {f'recall_at_{k}': 0.0 for k in self.k_values}
            
        results = {}
        for k in self.k_values:
            results[f'recall_at_{k}'] = self.recall_sums[k] / self.total_samples
            
        return results


class F1AtK:
    """
    F1@K metric combining Precision@K and Recall@K.
    """
    
    def __init__(self, k_values: List[int] = [1, 3, 5, 10]):
        self.precision_at_k = PrecisionAtK(k_values)
        self.recall_at_k = RecallAtK(k_values)
        
    def reset(self):
        """Reset metric state."""
        self.precision_at_k.reset()
        self.recall_at_k.reset()
        
    def update(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor
    ):
        """Update both precision and recall metrics."""
        self.precision_at_k.update(predictions, targets)
        self.recall_at_k.update(predictions, targets)
        
    def compute(self) -> Dict[str, float]:
        """Compute F1@k values."""
        precision_results = self.precision_at_k.compute()
        recall_results = self.recall_at_k.compute()
        
        f1_results = {}
        for k in self.precision_at_k.k_values:
            p_key = f'precision_at_{k}'
            r_key = f'recall_at_{k}'
            f1_key = f'f1_at_{k}'
            
            precision = precision_results[p_key]
            recall = recall_results[r_key]
            
            if precision + recall > 0:
                f1_results[f1_key] = 2 * precision * recall / (precision + recall)
            else:
                f1_results[f1_key] = 0.0
                
        return f1_results


class MeanAveragePrecision:
    """
    Mean Average Precision (mAP) for multi-label classification.
    
    Computes the mean of average precision scores for each sample.
    """
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        """Reset metric state."""
        self.total_samples = 0
        self.ap_sum = 0.0
        
    def _average_precision(
        self, 
        predictions: np.ndarray, 
        targets: np.ndarray
    ) -> float:
        """Compute average precision for a single sample."""
        # Get indices sorted by prediction scores (descending)
        sorted_indices = np.argsort(predictions)[::-1]
        sorted_targets = targets[sorted_indices]
        
        # Find positions of relevant items
        relevant_positions = np.where(sorted_targets > 0)[0]
        
        if len(relevant_positions) == 0:
            return 0.0
        
        # Compute precision at each relevant position
        precisions = []
        for pos in relevant_positions:
            # Precision at position pos+1 (1-indexed)
            precision = np.sum(sorted_targets[:pos+1]) / (pos + 1)
            precisions.append(precision)
        
        # Average precision is the mean of precisions at relevant positions
        return np.mean(precisions)
    
    def update(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor
    ):
        """
        Update metric with batch predictions and targets.
        
        Args:
            predictions: Model predictions [batch_size, num_labels]
            targets: Ground truth labels [batch_size, num_labels]
        """
        batch_size = predictions.size(0)
        self.total_samples += batch_size
        
        # Convert to numpy for easier processing
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
            
        for i in range(batch_size):
            ap = self._average_precision(predictions[i], targets[i])
            self.ap_sum += ap
            
    def compute(self) -> float:
        """Compute mean average precision."""
        if self.total_samples == 0:
            return 0.0
        return self.ap_sum / self.total_samples


class ExtremeMultiLabelMetrics:
    """
    Comprehensive metrics suite for extreme multi-label classification.
    
    Combines multiple metrics relevant to extreme classification scenarios
    with focus on top-k performance.
    """
    
    def __init__(
        self, 
        k_values: List[int] = [1, 3, 5, 10],
        compute_map: bool = True
    ):
        self.k_values = k_values
        self.compute_map = compute_map
        
        self.precision_at_k = PrecisionAtK(k_values)
        self.ndcg_at_k = NDCGAtK(k_values)
        self.recall_at_k = RecallAtK(k_values)
        self.f1_at_k = F1AtK(k_values)
        
        if compute_map:
            self.map_metric = MeanAveragePrecision()
            
        self.reset()
        
    def reset(self):
        """Reset all metrics."""
        self.precision_at_k.reset()
        self.ndcg_at_k.reset()
        self.recall_at_k.reset()
        self.f1_at_k.reset()
        
        if self.compute_map:
            self.map_metric.reset()
            
    def update(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor
    ):
        """Update all metrics with batch data."""
        self.precision_at_k.update(predictions, targets)
        self.ndcg_at_k.update(predictions, targets)
        self.recall_at_k.update(predictions, targets)
        self.f1_at_k.update(predictions, targets)
        
        if self.compute_map:
            self.map_metric.update(predictions, targets)
            
    def compute(self) -> Dict[str, float]:
        """Compute all metrics and return results."""
        results = {}
        
        # Add all k-based metrics
        results.update(self.precision_at_k.compute())
        results.update(self.ndcg_at_k.compute())
        results.update(self.recall_at_k.compute())
        results.update(self.f1_at_k.compute())
        
        # Add mAP if computed
        if self.compute_map:
            results['mean_average_precision'] = self.map_metric.compute()
            
        return results
    
    def compute_summary(self) -> Dict[str, float]:
        """Compute summary metrics focusing on key performance indicators."""
        all_results = self.compute()
        
        summary = {}
        
        # Focus on key metrics for extreme classification
        if 'precision_at_3' in all_results:
            summary['precision_at_3'] = all_results['precision_at_3']
        if 'ndcg_at_5' in all_results:
            summary['ndcg_at_5'] = all_results['ndcg_at_5']
        if 'precision_at_1' in all_results:
            summary['precision_at_1'] = all_results['precision_at_1']
        if 'precision_at_5' in all_results:
            summary['precision_at_5'] = all_results['precision_at_5']
        if 'mean_average_precision' in all_results:
            summary['mean_average_precision'] = all_results['mean_average_precision']
            
        # Compute macro averages for key metrics
        precision_values = [v for k, v in all_results.items() if k.startswith('precision_at_')]
        if precision_values:
            summary['macro_precision_at_k'] = np.mean(precision_values)
            
        ndcg_values = [v for k, v in all_results.items() if k.startswith('ndcg_at_')]
        if ndcg_values:
            summary['macro_ndcg_at_k'] = np.mean(ndcg_values)
            
        return summary


def evaluate_extreme_classification(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    k_values: List[int] = [1, 3, 5, 10],
    return_individual: bool = False
) -> Dict[str, float]:
    """
    Convenience function to evaluate extreme multi-label classification performance.
    
    Args:
        predictions: Model predictions [batch_size, num_labels]
        targets: Ground truth labels [batch_size, num_labels]
        k_values: List of k values to evaluate
        return_individual: Whether to return individual sample metrics
        
    Returns:
        Dictionary of computed metrics
    """
    metrics = ExtremeMultiLabelMetrics(k_values=k_values)
    metrics.update(predictions, targets)
    
    if return_individual:
        return metrics.compute()
    else:
        return metrics.compute_summary()