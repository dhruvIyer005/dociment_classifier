# model.py
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from typing import Optional, Dict, Any, Tuple
from config import config


class SciBERTDocumentClassifier(nn.Module):
    """
    SciBERT-based document classifier with chunk averaging.
    Processes documents in 400-450 token chunks and averages logits.
    """
    
    def __init__(self, model_name: str = None, num_labels: int = 5):
        super().__init__()
        
        if model_name is None:
            model_name = config.MODEL_NAME
        
        # Load SciBERT
        self.model_name = model_name
        self.num_labels = num_labels
        
        # Load pretrained SciBERT
        self.config = AutoConfig.from_pretrained(
            model_name,
            num_labels=num_labels,
            problem_type="single_label_classification"
        )
        self.bert = AutoModel.from_pretrained(model_name, config=self.config)
        
        # Classification head
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Forward pass with optional labels for training.
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            labels: Optional labels for loss computation
            
        Returns:
            Dictionary with logits and optional loss
        """
        
        # Get BERT outputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Use [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0]  # [CLS] token
        pooled_output = self.dropout(pooled_output)
        
        # Get logits
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
        
        return {
            "loss": loss,
            "logits": logits,
        }
    
    def predict_single(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[int, float]:
        """
        Predict single sample (for chunk averaging).
        
        Returns:
            (predicted_class_id, confidence)
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            logits = outputs["logits"]
            
            # Apply temperature scaling for calibration (T=1.5)
            temperature = 1.5
            scaled_logits = logits / temperature
            probs = torch.softmax(scaled_logits, dim=-1)
            
            pred_id = torch.argmax(probs, dim=-1).item()
            confidence = probs[0, pred_id].item()
            
        return pred_id, confidence


def create_model(model_name: str = None, num_labels: int = 5, device: str = "cuda"):
    """
    Create and load SciBERT classifier.
    
    Args:
        model_name: Model identifier (default: allenai/scibert_scivocab_uncased)
        num_labels: Number of classification labels
        device: Device to load model on ("cuda" or "cpu")
        
    Returns:
        Loaded model on specified device
    """
    model = SciBERTDocumentClassifier(model_name, num_labels)
    model = model.to(device)
    return model