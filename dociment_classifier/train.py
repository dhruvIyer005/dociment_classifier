"""
train.py - Simple training script for Legal-BERT (GPU optimized)
Trains on real PDFs from data/{ACM,IEEE,Springer,Legal,Compliance}/ directories
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
import numpy as np
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, DataCollatorWithPadding,
    EarlyStoppingCallback
)
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from config import (
    DATA_DIR, MODEL_PATH, MODEL_NAME, LABELS, LABEL2ID, ID2LABEL,
    NUM_LABELS, MAX_LENGTH, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE,
    USE_FP16, DEVICE
)
from text_processor import TextProcessor

print("="*70)
print("Legal-BERT Training - GPU Optimized")
print("="*70)
print(f"Device: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print()


def load_pdf_texts():
    """Load all PDFs from data/ directories"""
    texts = []
    labels_list = []
    
    print(f"Loading PDFs from {DATA_DIR}")
    
    for label in LABELS:
        label_dir = DATA_DIR / label
        
        if not label_dir.exists():
            print(f"  ⚠ Directory not found: {label_dir}")
            continue
        
        pdf_files = list(label_dir.glob("*.pdf"))
        print(f"  {label}: {len(pdf_files)} PDFs")
        
        for pdf_path in pdf_files:
            try:
                from text_processor import TextProcessor as TP
                processor = TP(AutoTokenizer.from_pretrained(MODEL_NAME))
                text = processor.extract_text(str(pdf_path))
                
                if text:
                    texts.append(text)
                    labels_list.append(LABEL2ID[label])
                    print(f"    ✓ {pdf_path.name}")
                else:
                    print(f"    ✗ {pdf_path.name} (empty)")
                    
            except Exception as e:
                print(f"    ✗ {pdf_path.name} ({e})")
    
    print(f"\nTotal documents loaded: {len(texts)}")
    return texts, labels_list


def compute_metrics(eval_pred):
    """Compute evaluation metrics"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted"
    )
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def main():
    """Main training loop"""
    
    # Load data
    print("\n[1] Loading PDF data...")
    texts, labels = load_pdf_texts()
    
    if len(texts) < 10:
        print("ERROR: Not enough PDFs for training!")
        return
    
    # Split data
    print("\n[2] Splitting data...")
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        test_texts, test_labels, test_size=0.5, random_state=42
    )
    
    print(f"  Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}")
    
    # Load tokenizer
    print("\n[3] Loading SciBERT tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Create datasets
    print("[4] Creating datasets...")
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
        )
    
    train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels})
    val_dataset = Dataset.from_dict({"text": val_texts, "label": val_labels})
    test_dataset = Dataset.from_dict({"text": test_texts, "label": test_labels})
    
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)
    
    # Load model
    print("[5] Loading SciBERT model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_LABELS
    )
    
    # Training arguments
    print("[6] Setting up training...")
    training_args = TrainingArguments(
        output_dir=str(MODEL_PATH),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        warmup_ratio=0.1,
        weight_decay=0.01,
        fp16=USE_FP16,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_steps=10,
        seed=42,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    # Train
    print("\n[7] Starting training...")
    trainer.train()
    
    # Evaluate
    print("\n[8] Evaluating on test set...")
    test_results = trainer.evaluate(eval_dataset=test_dataset)
    print(f"  Accuracy: {test_results['eval_accuracy']:.4f}")
    print(f"  F1: {test_results['eval_f1']:.4f}")
    
    # Save
    print(f"\n[9] Saving model to {MODEL_PATH}...")
    model.save_pretrained(MODEL_PATH)
    tokenizer.save_pretrained(MODEL_PATH)
    
    print("\n✓ Training complete!")


if __name__ == "__main__":
    main()
