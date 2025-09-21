"""
Example usage and training script for Enhanced ExectModel

This script demonstrates how to use the enhanced ExectModel for extreme 
multi-label classification with focus on Precision@3 and NDCG@5 optimization.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple
import argparse
import json
import os
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup

from extreme_classification import (
    ExectModel, 
    ExtremeMultiLabelMetrics,
    evaluate_extreme_classification
)


class ExtremeMultiLabelDataset(Dataset):
    """
    Dataset class for extreme multi-label classification.
    """
    
    def __init__(
        self,
        texts: List[str],
        labels: List[List[int]],
        tokenizer,
        max_length: int = 512,
        num_labels: int = 10000
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_labels = num_labels
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        labels = self.labels[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Create multi-hot label vector
        label_vector = torch.zeros(self.num_labels)
        for label_id in labels:
            if 0 <= label_id < self.num_labels:
                label_vector[label_id] = 1.0
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': label_vector
        }


def create_synthetic_data(
    num_samples: int = 1000,
    num_labels: int = 10000,
    avg_labels_per_sample: int = 3,
    vocab_size: int = 1000
) -> Tuple[List[str], List[List[int]]]:
    """
    Create synthetic data for demonstration and testing.
    
    Args:
        num_samples: Number of samples to generate
        num_labels: Total number of possible labels
        avg_labels_per_sample: Average number of labels per sample
        vocab_size: Vocabulary size for text generation
        
    Returns:
        Tuple of (texts, labels)
    """
    texts = []
    labels = []
    
    # Create Zipfian distribution for label frequencies (realistic for extreme classification)
    label_probs = np.array([1/i for i in range(1, num_labels + 1)])
    label_probs = label_probs / label_probs.sum()
    
    for i in range(num_samples):
        # Generate synthetic text
        text_length = np.random.randint(10, 100)
        words = np.random.randint(0, vocab_size, text_length)
        text = ' '.join([f'word_{w}' for w in words])
        texts.append(text)
        
        # Generate labels with realistic distribution
        num_labels_sample = max(1, np.random.poisson(avg_labels_per_sample))
        sample_labels = np.random.choice(
            num_labels, 
            size=min(num_labels_sample, num_labels), 
            replace=False,
            p=label_probs
        ).tolist()
        labels.append(sample_labels)
    
    return texts, labels


def train_epoch(
    model: ExectModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    device: torch.device = torch.device('cpu'),
    gradient_accumulation_steps: int = 1,
    max_grad_norm: float = 1.0
) -> Dict[str, float]:
    """Train model for one epoch."""
    model.train()
    
    total_loss = 0.0
    total_focused_loss = 0.0
    total_contrastive_loss = 0.0
    num_batches = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch_idx, batch in enumerate(progress_bar):
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs['loss'] / gradient_accumulation_steps
        focused_loss = outputs.get('focused_loss', 0.0)
        contrastive_loss = outputs.get('contrastive_loss', 0.0)
        
        # Backward pass
        loss.backward()
        
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            if scheduler:
                scheduler.step()
            optimizer.zero_grad()
        
        # Update metrics
        total_loss += loss.item() * gradient_accumulation_steps
        total_focused_loss += focused_loss.item() if isinstance(focused_loss, torch.Tensor) else focused_loss
        total_contrastive_loss += contrastive_loss.item() if isinstance(contrastive_loss, torch.Tensor) else contrastive_loss
        num_batches += 1
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': total_loss / num_batches,
            'focused': total_focused_loss / num_batches,
            'contrastive': total_contrastive_loss / num_batches
        })
    
    return {
        'train_loss': total_loss / num_batches,
        'train_focused_loss': total_focused_loss / num_batches,
        'train_contrastive_loss': total_contrastive_loss / num_batches
    }


def evaluate_model(
    model: ExectModel,
    dataloader: DataLoader,
    device: torch.device = torch.device('cpu'),
    k_values: List[int] = [1, 3, 5, 10]
) -> Dict[str, float]:
    """Evaluate model on validation/test set."""
    model.eval()
    
    metrics = ExtremeMultiLabelMetrics(k_values=k_values)
    total_loss = 0.0
    num_batches = 0
    
    progress_bar = tqdm(dataloader, desc="Evaluating")
    
    with torch.no_grad():
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs['loss']
            logits = outputs['logits']
            
            # Update loss
            total_loss += loss.item()
            num_batches += 1
            
            # Update metrics
            metrics.update(logits, labels)
            
            # Update progress bar
            progress_bar.set_postfix({'val_loss': total_loss / num_batches})
    
    # Compute final metrics
    metric_results = metrics.compute_summary()
    metric_results['val_loss'] = total_loss / num_batches
    
    return metric_results


def main():
    parser = argparse.ArgumentParser(description='Train Enhanced ExectModel')
    parser.add_argument('--num_labels', type=int, default=1000, help='Number of labels')
    parser.add_argument('--num_samples', type=int, default=5000, help='Number of training samples')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--warmup_steps', type=int, default=1000, help='Warmup steps')
    parser.add_argument('--max_length', type=int, default=128, help='Max sequence length')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--use_synthetic_data', action='store_true', help='Use synthetic data')
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained('hfl/chinese-macbert-base')
    
    # Create or load data
    if args.use_synthetic_data:
        print("Creating synthetic data...")
        texts, labels = create_synthetic_data(
            num_samples=args.num_samples,
            num_labels=args.num_labels,
            avg_labels_per_sample=3
        )
        
        # Split into train/val
        split_idx = int(0.8 * len(texts))
        train_texts, val_texts = texts[:split_idx], texts[split_idx:]
        train_labels, val_labels = labels[:split_idx], labels[split_idx:]
    else:
        print("Please implement your data loading logic here")
        return
    
    # Create datasets
    train_dataset = ExtremeMultiLabelDataset(
        train_texts, train_labels, tokenizer, 
        args.max_length, args.num_labels
    )
    val_dataset = ExtremeMultiLabelDataset(
        val_texts, val_labels, tokenizer,
        args.max_length, args.num_labels
    )
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False
    )
    
    # Initialize model
    print("Initializing Enhanced ExectModel...")
    model = ExectModel(
        num_labels=args.num_labels,
        bert_model_name='hfl/chinese-macbert-base',
        tail_threshold=0.01,
        use_tail_attention=True,
        dropout_rate=0.1
    ).to(device)
    
    # Setup optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    
    total_steps = len(train_dataloader) * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps
    )
    
    # Training loop
    best_ndcg5 = 0.0
    training_history = []
    
    print("Starting training...")
    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        
        # Train
        train_metrics = train_epoch(
            model, train_dataloader, optimizer, scheduler, device
        )
        
        # Evaluate
        val_metrics = evaluate_model(
            model, val_dataloader, device, k_values=[1, 3, 5, 10]
        )
        
        # Combine metrics
        epoch_metrics = {**train_metrics, **val_metrics}
        training_history.append(epoch_metrics)
        
        # Print results
        print(f"Train Loss: {train_metrics['train_loss']:.4f}")
        print(f"Val Loss: {val_metrics['val_loss']:.4f}")
        print(f"Precision@3: {val_metrics.get('precision_at_3', 0.0):.4f}")
        print(f"NDCG@5: {val_metrics.get('ndcg_at_5', 0.0):.4f}")
        print(f"Precision@1: {val_metrics.get('precision_at_1', 0.0):.4f}")
        
        # Save best model
        current_ndcg5 = val_metrics.get('ndcg_at_5', 0.0)
        if current_ndcg5 > best_ndcg5:
            best_ndcg5 = current_ndcg5
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model.pt'))
            print(f"New best model saved with NDCG@5: {best_ndcg5:.4f}")
    
    # Save final model and training history
    torch.save(model.state_dict(), os.path.join(args.output_dir, 'final_model.pt'))
    
    with open(os.path.join(args.output_dir, 'training_history.json'), 'w') as f:
        json.dump(training_history, f, indent=2)
    
    print(f"\nTraining completed! Best NDCG@5: {best_ndcg5:.4f}")
    print(f"Models and logs saved to {args.output_dir}")


if __name__ == "__main__":
    main()