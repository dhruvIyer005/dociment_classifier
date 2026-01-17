"""
classifier.py - Legal-BERT document classifier
Loads nlpaueb/legal-bert-base-uncased for single-label classification.
GPU + fp16 optimized. No DistilBERT code.
"""

import torch
import torch.nn.functional as F
import logging
from pathlib import Path
from typing import Dict, Tuple, List
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from config import (
    MODEL_NAME, MODEL_PATH, DEVICE, TEMPERATURE,
    ID2LABEL, LABELS, NUM_LABELS, USE_FP16
)
from text_processor import TextProcessor

logger = logging.getLogger(__name__)


class DocumentClassifier:
    """Legal-BERT classifier with chunk-based inference (GPU + fp16)"""
    
    def __init__(self, model_path: str = None):
        """Load Legal-BERT model and tokenizer"""
        
        if model_path is None:
            model_path = str(MODEL_PATH)
        
        self.device = DEVICE
        self.use_fp16 = USE_FP16 and self.device == "cuda"
        logger.info(f"Using device: {self.device}, fp16: {self.use_fp16}")
        
        try:
            # Load from saved path
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            logger.info(f"✓ Loaded model from {model_path}")
        except Exception as e:
            logger.warning(f"Could not load from {model_path}: {e}")
            logger.info(f"Loading base Legal-BERT (nlpaueb/legal-bert-base-uncased)...")
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                MODEL_NAME, num_labels=NUM_LABELS
            )
            logger.info(f"✓ Loaded {MODEL_NAME}")
        
        self.model = self.model.to(self.device)
        
        # Enable fp16 if available
        if self.use_fp16:
            self.model = self.model.half()
            logger.info("✓ Enabled fp16 precision")
        
        self.model.eval()
        
        self.text_processor = TextProcessor(self.tokenizer)
    
    def predict(self, text: str) -> Tuple[str, float]:
        """
        Predict class for document text.
        Uses chunk-based inference with logit averaging.
        
        Returns:
            (predicted_label, confidence)
        """
        # Extract and chunk text
        chunks = self.text_processor.chunk_text(text)
        
        if not chunks:
            logger.warning("No valid chunks generated")
            return "Unknown", 0.0
        
        # Collect logits from all chunks
        all_logits = []
        
        with torch.no_grad():
            for input_ids, attention_mask in chunks:
                input_ids = input_ids.unsqueeze(0).to(self.device)
                attention_mask = attention_mask.unsqueeze(0).to(self.device)
                
                # Use fp16 if enabled
                if self.use_fp16:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(input_ids, attention_mask)
                        logits = outputs.logits
                else:
                    outputs = self.model(input_ids, attention_mask)
                    logits = outputs.logits
                
                all_logits.append(logits.cpu().float())
        
        # Average logits across all chunks
        avg_logits = torch.mean(torch.cat(all_logits, dim=0), dim=0, keepdim=True)
        
        # Temperature scaling for calibration
        scaled_logits = avg_logits / TEMPERATURE
        probs = F.softmax(scaled_logits, dim=-1)
        
        # Get prediction
        pred_id = torch.argmax(probs, dim=-1).item()
        confidence = probs[0, pred_id].item()
        pred_label = ID2LABEL.get(pred_id, "Unknown")
        
        return pred_label, confidence
    
    def batch_predict(self, pdf_paths: List[str]) -> List[Dict]:
        """
        Predict multiple PDFs.
        
        Returns:
            List of dicts with filename, label, confidence
        """
        results = []
        
        for pdf_path in pdf_paths:
            try:
                # Extract text
                text = self.text_processor.extract_text(pdf_path)
                
                if not text:
                    results.append({
                        "filename": Path(pdf_path).name,
                        "label": "Error",
                        "confidence": 0.0,
                        "error": "Failed to extract text"
                    })
                    continue
                
                # Predict
                label, confidence = self.predict(text)
                
                results.append({
                    "filename": Path(pdf_path).name,
                    "label": label,
                    "confidence": round(confidence, 4),
                })
                
            except Exception as e:
                logger.error(f"Error processing {pdf_path}: {e}")
                results.append({
                    "filename": Path(pdf_path).name,
                    "label": "Error",
                    "confidence": 0.0,
                    "error": str(e)
                })
        
        return results


def create_classifier(model_path: str = None) -> DocumentClassifier:
    """Factory function to create classifier"""
    return DocumentClassifier(model_path)
