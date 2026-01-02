# model.py
import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertConfig, DistilBertPreTrainedModel
from typing import Optional, Dict, Any
from config import config


class DistilBertDocumentClassifier(DistilBertPreTrainedModel):
 
    
    def __init__(self, config: DistilBertConfig):
        super().__init__(config)
        
        # DistilBERT base model
        self.distilbert = DistilBertModel(config)
        
        # Classification head
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        
        # Initialize weights
        self.post_init()
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Dict[str, Any]:
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Get DistilBERT outputs
        outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=return_dict,
        )
        
        # Get [CLS] token representation
        sequence_output = outputs[0]  # Last hidden state
        pooled_output = sequence_output[:, 0]  # [CLS] token
        pooled_output = self.dropout(pooled_output)
        
        # Classification logits
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
        
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output
        
        return {
            "loss": loss,
            "logits": logits,
            "hidden_states": outputs.hidden_states,
            "attentions": outputs.attentions,
        }


class SimpleDocumentClassifier(nn.Module):

    def __init__(self, num_classes: int = 5):
        super().__init__()
        
        # Load DistilBERT
        self.distilbert = DistilBertModel.from_pretrained(config.MODEL_NAME)
        
        # Freeze some layers for small dataset (optional)
        self._freeze_layers()
        
        # Classification head
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, num_classes)
        
    def _freeze_layers(self):
        """Freeze early layers to prevent overfitting on small dataset"""
        # Freeze first 3 layers (out of 6 in DistilBERT)
        for param in self.distilbert.embeddings.parameters():
            param.requires_grad = False
            
        for layer in self.distilbert.transformer.layer[:3]:
            for param in layer.parameters():
                param.requires_grad = False
    
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor, 
        labels: Optional[torch.Tensor] = None
    ):
        # Get DistilBERT outputs
        outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        
        # [CLS] token
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        
        # Classification logits
        logits = self.classifier(pooled_output)
        
        # Calculate loss if labels provided
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return {"loss": loss, "logits": logits}
        
        return {"logits": logits}
    
    def predict_proba(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Get probability predictions"""
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            logits = outputs["logits"]
            return torch.softmax(logits, dim=-1)
    
    def predict(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Get class predictions"""
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            logits = outputs["logits"]
            return torch.argmax(logits, dim=-1)


def create_model(
    model_type: str = "simple",
    num_classes: Optional[int] = None,
    use_pretrained_config: bool = True
) -> nn.Module:
    """
    Factory function to create model
    
    Args:
        model_type: "simple" or "pretrained"
        num_classes: Number of output classes (defaults to config.LABELS)
        use_pretrained_config: Use config from pretrained model
    
    Returns:
        Initialized model
    """
    if num_classes is None:
        num_classes = len(config.LABELS)
    
    if model_type == "pretrained":
        # Use DistilBertPreTrainedModel approach
        model_config = DistilBertConfig.from_pretrained(
            config.MODEL_NAME,
            num_labels=num_classes,
            id2label=config.ID2LABEL,
            label2id=config.LABEL2ID,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
        )
        model = DistilBertDocumentClassifier.from_pretrained(
            config.MODEL_NAME,
            config=model_config,
            ignore_mismatched_sizes=True,
        )
    else:
        # Use simple approach
        model = SimpleDocumentClassifier(num_classes=num_classes)
    
    return model


# For backward compatibility
DocumentClassifier = SimpleDocumentClassifier