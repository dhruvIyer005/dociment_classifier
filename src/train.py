# train.py
import os
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer, 
    Trainer, 
    TrainingArguments, 
    EvalPrediction,
    EarlyStoppingCallback
)
from datasets import Dataset, DatasetDict
import evaluate
from model import create_model
from config import config
from pdf_processor import PDFDatasetBuilder


class ClassificationTrainer:
    """Train DistilBERT classifier for document classification"""
    
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.trainer = None
        
    def load_and_prepare_data(self) -> Tuple[DatasetDict, Dict]:
        """
        Load and prepare training data from real PDFs and synthetic data
        
        Returns:
            DatasetDict with train/val/test splits
            Label mapping dictionary
        """
        print("Loading and preparing data...")
        
        # 1. Load real PDF data
        pdf_builder = PDFDatasetBuilder()
        real_df = pdf_builder.process_pdfs_by_label(str(config.RAW_PDFS_DIR))
        
        if not real_df.empty:
            print(f"Loaded {len(real_df)} real PDF examples")
        else:
            print("No real PDF data found")
            real_df = pd.DataFrame()
        
        # 2. Load synthetic data
        synthetic_path = config.SYNTHETIC_DATA_DIR / "synthetic_training_data.csv"
        if synthetic_path.exists():
            synthetic_df = pd.read_csv(synthetic_path)
            print(f"Loaded {len(synthetic_df)} synthetic examples")
        else:
            print("No synthetic data found, generating...")
            from generate_data import SyntheticDataGenerator
            generator = SyntheticDataGenerator()
            synthetic_df = generator.generate_examples(examples_per_class=15)
        
        # 3. Combine data
        if not real_df.empty and not synthetic_df.empty:
            # Ensure both have required columns
            if 'label_id' not in real_df.columns and 'label' in real_df.columns:
                real_df['label_id'] = real_df['label'].map(config.LABEL2ID)
            
            if 'label_id' not in synthetic_df.columns and 'label' in synthetic_df.columns:
                synthetic_df['label_id'] = synthetic_df['label'].map(config.LABEL2ID)
            
            # Combine
            combined_df = pd.concat([real_df, synthetic_df], ignore_index=True)
            print(f"Combined dataset: {len(combined_df)} examples")
        elif not real_df.empty:
            combined_df = real_df
        else:
            combined_df = synthetic_df
        
        # 4. Prepare text and labels
        texts = combined_df['text'].astype(str).tolist()
        labels = combined_df['label_id'].astype(int).tolist()
        
        # 5. Split data
        train_texts, temp_texts, train_labels, temp_labels = train_test_split(
            texts, labels, 
            test_size=config.VAL_RATIO + config.TEST_RATIO,
            stratify=labels,
            random_state=42
        )
        
        val_texts, test_texts, val_labels, test_labels = train_test_split(
            temp_texts, temp_labels,
            test_size=config.TEST_RATIO / (config.VAL_RATIO + config.TEST_RATIO),
            stratify=temp_labels,
            random_state=42
        )
        
        print(f"Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}")
        
        # 6. Create datasets
        train_dataset = self._create_dataset(train_texts, train_labels)
        val_dataset = self._create_dataset(val_texts, val_labels)
        test_dataset = self._create_dataset(test_texts, test_labels)
        
        dataset_dict = DatasetDict({
            'train': train_dataset,
            'validation': val_dataset,
            'test': test_dataset
        })
        
        return dataset_dict, config.ID2LABEL
    
    def _create_dataset(self, texts: List[str], labels: List[int]) -> Dataset:
        """Create HuggingFace dataset from texts and labels"""
        return Dataset.from_dict({
            'text': texts,
            'label': labels
        })
    
    def _tokenize_function(self, examples):
        """Tokenize text examples"""
        return self.tokenizer(
            examples['text'],
            truncation=True,
            padding=True,
            max_length=config.MAX_LENGTH,
            return_tensors=None  # Let Trainer handle batching
        )
    
    def _compute_metrics(self, eval_pred: EvalPrediction):
        """Compute evaluation metrics"""
        accuracy_metric = evaluate.load("accuracy")
        f1_metric = evaluate.load("f1")
        
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
        f1 = f1_metric.compute(predictions=predictions, references=labels, average="weighted")
        
        return {**accuracy, **f1}
    
    def initialize_model(self, num_classes: Optional[int] = None):
        """Initialize model and tokenizer"""
        if num_classes is None:
            num_classes = len(config.LABELS)
        
        print(f"Initializing model with {num_classes} classes...")
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
        
        # Initialize model
        self.model = create_model(
            model_type="simple",
            num_classes=num_classes,
            use_pretrained_config=True
        )
        
        print(f"Model initialized: {config.MODEL_NAME}")
    
    def prepare_datasets(self, dataset_dict: DatasetDict) -> DatasetDict:
        """Tokenize datasets"""
        print("Tokenizing datasets...")
        
        tokenized_datasets = dataset_dict.map(
            self._tokenize_function,
            batched=True,
            remove_columns=['text']  # Remove original text column
        )
        
        return tokenized_datasets
    
    def train(self, 
              dataset_dict: DatasetDict,
              output_dir: Optional[str] = None,
              use_early_stopping: bool = True):
        """Train the model"""
        
        if output_dir is None:
            output_dir = str(config.MODEL_SAVE_PATH)
        
        print(f"Training model, saving to: {output_dir}")
        
        # Prepare training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=config.NUM_EPOCHS,
            per_device_train_batch_size=config.BATCH_SIZE,
            per_device_eval_batch_size=config.BATCH_SIZE,
            warmup_ratio=config.WARMUP_RATIO,
            weight_decay=config.WEIGHT_DECAY,
            learning_rate=config.LEARNING_RATE,
            logging_dir=str(config.OUTPUTS_DIR / "logs"),
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            report_to="none",
            seed=42,
        )
        
        # Prepare callbacks
        callbacks = []
        if use_early_stopping:
            callbacks.append(EarlyStoppingCallback(
                early_stopping_patience=config.PATIENCE,
                early_stopping_threshold=config.MIN_DELTA
            ))
        
        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset_dict['train'],
            eval_dataset=dataset_dict['validation'],
            tokenizer=self.tokenizer,
            compute_metrics=self._compute_metrics,
            callbacks=callbacks,
        )
        
        # Train model
        print("Starting training...")
        train_result = self.trainer.train()
        
        # Save model using trainer (handles both safetensors and pytorch)
        print(f"Saving model to {output_dir}...")
        self.trainer.save_model(output_dir)
        # Also explicitly save model state with torch to ensure all weights are saved
        import torch
        torch.save(self.model.state_dict(), Path(output_dir) / "pytorch_model.bin")
        self.tokenizer.save_pretrained(output_dir)
        
        # Save training metrics
        metrics_path = Path(output_dir) / "training_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(train_result.metrics, f, indent=2)
        
        # Save label mapping
        label_mapping = {
            "id2label": config.ID2LABEL,
            "label2id": config.LABEL2ID
        }
        mapping_path = Path(output_dir) / "label_mapping.json"
        with open(mapping_path, 'w') as f:
            json.dump(label_mapping, f, indent=2)
        
        print(f"Model saved to: {output_dir}")
        
        return train_result
    
    def evaluate(self, dataset_dict: DatasetDict):
        """Evaluate model on test set"""
        if self.trainer is None:
            print("Model not trained yet")
            return
        
        print("Evaluating on test set...")
        eval_results = self.trainer.evaluate(dataset_dict['test'])
        
        # Print results
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        for metric, value in eval_results.items():
            print(f"{metric}: {value:.4f}")
        
        # Save evaluation results
        eval_path = config.OUTPUTS_DIR / "evaluation_results.json"
        with open(eval_path, 'w') as f:
            json.dump(eval_results, f, indent=2)
        
        print(f"Evaluation results saved to: {eval_path}")
        
        return eval_results


def train_classifier_from_dataframe(df: pd.DataFrame, output_dir: str):
    """
    Legacy function for backward compatibility
    Train classifier directly from DataFrame
    """
    trainer = ClassificationTrainer()
    
    # Prepare data
    if 'label_id' not in df.columns and 'label' in df.columns:
        df['label_id'] = df['label'].map(config.LABEL2ID)
    
    texts = df['text'].astype(str).tolist()
    labels = df['label_id'].astype(int).tolist()
    
    # Create dataset
    dataset = trainer._create_dataset(texts, labels)
    dataset_dict = DatasetDict({
        'train': dataset,
        'validation': dataset,  # Same for validation (simplified)
        'test': dataset
    })
    
    # Train
    trainer.initialize_model()
    tokenized_datasets = trainer.prepare_datasets(dataset_dict)
    train_result = trainer.train(tokenized_datasets, output_dir)
    
    return trainer.model, trainer.trainer


def main():
    """Main training pipeline"""
    print("="*60)
    print("DISTILBERT DOCUMENT CLASSIFIER TRAINING")
    print("="*60)
    
    # Initialize trainer
    trainer = ClassificationTrainer()
    
    # Load and prepare data
    dataset_dict, label_mapping = trainer.load_and_prepare_data()
    
    # Check if we have enough data
    total_examples = sum(len(dataset_dict[split]) for split in dataset_dict)
    if total_examples < 20:
        print(f"Warning: Very small dataset ({total_examples} examples)")
        print("Consider adding more real PDFs or generating more synthetic data")
    
    # Initialize model
    trainer.initialize_model()
    
    # Prepare datasets
    tokenized_datasets = trainer.prepare_datasets(dataset_dict)
    
    # Train model
    train_result = trainer.train(tokenized_datasets)
    
    # Evaluate on test set
    if 'test' in tokenized_datasets:
        trainer.evaluate(tokenized_datasets)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Model saved to: {config.MODEL_SAVE_PATH}")
    print(f"Labels: {config.LABELS}")
    
    return trainer


if __name__ == "__main__":
    # Train with full pipeline
    trainer = main()