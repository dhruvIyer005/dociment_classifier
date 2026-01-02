# config.py
import os
from pathlib import Path
from typing import Dict, List, Optional

class ClassifierConfig:
    def __init__(self):
        self.BASE_DIR = Path(__file__).parent.parent
        self._setup_paths()
        self._setup_model_config()
        self._setup_labels()
        
    def _setup_paths(self):
        """Setup all project paths"""
        self.DATA_DIR = self.BASE_DIR / "data"
        self.RAW_PDFS_DIR = self.DATA_DIR / "raw_pdf"  # Your 25 real PDFs
        self.SYNTHETIC_DATA_DIR = self.DATA_DIR / "synthetic"  # Generated CSV data
        self.MODELS_DIR = self.BASE_DIR / "models"
        self.OUTPUTS_DIR = self.BASE_DIR / "outputs"
        
        # Training/validation splits
        self.TRAIN_DATA_PATH = self.DATA_DIR / "train_data.csv"
        self.VAL_DATA_PATH = self.DATA_DIR / "val_data.csv"
        self.TEST_DATA_PATH = self.DATA_DIR / "test_data.csv"
        
        # Model paths
        self.MODEL_SAVE_PATH = self.MODELS_DIR / "distilbert_classifier"
        self.TOKENIZER_SAVE_PATH = self.MODELS_DIR / "tokenizer"
        
        # Bulk prediction output
        self.BULK_PREDICTIONS_PATH = self.OUTPUTS_DIR / "bulk_predictions.csv"
        
        # Create directories
        for dir_path in [self.DATA_DIR, self.RAW_PDFS_DIR, self.SYNTHETIC_DATA_DIR,
                        self.MODELS_DIR, self.OUTPUTS_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def _setup_model_config(self):
        """Setup DistilBERT model and training parameters"""
        # Model
        self.MODEL_NAME = "distilbert-base-uncased"
        self.MAX_LENGTH = 512
        self.TRUNCATION = True
        self.PADDING = "max_length"
        
        # Training
        self.BATCH_SIZE = 8  # Adjust based on your GPU memory
        self.NUM_EPOCHS = 10
        self.LEARNING_RATE = 2e-5
        self.WEIGHT_DECAY = 0.01
        self.WARMUP_RATIO = 0.1
        
        # Early stopping
        self.PATIENCE = 3
        self.MIN_DELTA = 0.001
        
        # Data split
        self.TRAIN_RATIO = 0.7
        self.VAL_RATIO = 0.15
        self.TEST_RATIO = 0.15
        
        # For small dataset (25 real + 75 synthetic)
        self.NUM_SYNTHETIC_SAMPLES = 75  # Your generated data
        self.REAL_SAMPLES_PER_CLASS = 5  # ~25 total across 5 classes
        
    def _setup_labels(self):
        """Setup classification labels - simplified for your use case"""
        # Your target categories
        self.LABELS = ["ieee", "springer", "acm", "compliance", "legal"]
        self.LABEL2ID = {label: idx for idx, label in enumerate(self.LABELS)}
        self.ID2LABEL = {idx: label for label, idx in self.LABEL2ID.items()}
        self.NUM_LABELS = len(self.LABELS)
        
        # Text extraction settings
        self.MIN_TEXT_LENGTH = 100  # Minimum characters to consider
        self.EXTRACT_FIRST_N_CHARS = 2000  # Extract first N chars for classification
        
    def get_pdf_paths(self) -> List[Path]:
        """Get all PDF file paths from raw_pdfs directory"""
        pdf_paths = []
        if self.RAW_PDFS_DIR.exists():
            # Assuming structure: raw_pdfs/{label}/*.pdf
            for label in self.LABELS:
                label_dir = self.RAW_PDFS_DIR / label
                if label_dir.exists():
                    pdf_paths.extend(label_dir.glob("*.pdf"))
        return pdf_paths
    
    def get_synthetic_data_path(self) -> Optional[Path]:
        """Get synthetic data CSV path"""
        csv_files = list(self.SYNTHETIC_DATA_DIR.glob("*.csv"))
        return csv_files[0] if csv_files else None
    
    def get_model_config(self) -> Dict:
        """Get model configuration for training"""
        return {
            "model_name": self.MODEL_NAME,
            "num_labels": self.NUM_LABELS,
            "id2label": self.ID2LABEL,
            "label2id": self.LABEL2ID,
            "attention_probs_dropout_prob": 0.1,
            "hidden_dropout_prob": 0.1,
        }
    
    def get_training_args(self, output_dir: Optional[str] = None) -> Dict:
        """Get training arguments"""
        if output_dir is None:
            output_dir = str(self.MODEL_SAVE_PATH)
            
        return {
            "output_dir": output_dir,
            "num_train_epochs": self.NUM_EPOCHS,
            "per_device_train_batch_size": self.BATCH_SIZE,
            "per_device_eval_batch_size": self.BATCH_SIZE,
            "warmup_ratio": self.WARMUP_RATIO,
            "weight_decay": self.WEIGHT_DECAY,
            "logging_dir": str(self.OUTPUTS_DIR / "logs"),
            "logging_steps": 10,
            "evaluation_strategy": "epoch",
            "save_strategy": "epoch",
            "load_best_model_at_end": True,
            "metric_for_best_model": "accuracy",
            "greater_is_better": True,
            "report_to": "none",  # Change to "wandb" if using weights & biases
        }

# Global config instance
config = ClassifierConfig()

# Export commonly used variables for backward compatibility
MODEL_NAME = config.MODEL_NAME
MAX_LENGTH = config.MAX_LENGTH
BATCH_SIZE = config.BATCH_SIZE
NUM_EPOCHS = config.NUM_EPOCHS
LEARNING_RATE = config.LEARNING_RATE
MODEL_SAVE_PATH = str(config.MODEL_SAVE_PATH)
LABELS = config.LABELS
LABEL2ID = config.LABEL2ID
ID2LABEL = config.ID2LABEL