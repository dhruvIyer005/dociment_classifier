# predict.py
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Union
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from config import config
from pdf_processor import PDFTextExtractor
import json 

class DocumentClassifierPredictor:
    """DistilBERT document classifier for bulk prediction"""
    
    def __init__(self, model_dir: Optional[str] = None):
        """
        Initialize predictor
        
        Args:
            model_dir: Directory containing trained model and tokenizer
                      If None, uses config.MODEL_SAVE_PATH
        """
        if model_dir is None:
            model_dir = str(config.MODEL_SAVE_PATH)
        
        self.model_dir = Path(model_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model and tokenizer
        self.model = None
        self.tokenizer = None
        self.label_mapping = None
        
        self._load_model()
    
    def _load_model(self):
        """Load trained model, tokenizer, and label mappings"""
        try:
            print(f"Loading model from: {self.model_dir}")
            
            # Load label mapping first to know how many labels
            mapping_path = self.model_dir / "label_mapping.json"
            if mapping_path.exists():
                with open(mapping_path, 'r') as f:
                    mapping = json.load(f)
                id2label = mapping.get("id2label", {})
                num_labels = len(id2label)
            else:
                num_labels = len(config.LABELS)
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, trust_remote_code=True)
            print("Loaded tokenizer")
            
            # Load model with proper num_labels config
            # NOTE: from_pretrained will load model.safetensors if available, otherwise pytorch_model.bin
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_dir,
                trust_remote_code=True,
                local_files_only=True,
                num_labels=num_labels
            )
            print("Loaded model architecture and weights")
            
            self.model.to(self.device)
            self.model.eval()
            print("Model ready for inference")
            
            # Load label mapping - uses string keys consistently
            mapping_path = self.model_dir / "label_mapping.json"
            if mapping_path.exists():
                with open(mapping_path, 'r') as f:
                    mapping = json.load(f)
                
                # id2label from JSON always has string keys
                self.id2label = mapping.get("id2label", {})
                self.label2id = mapping.get("label2id", {})
                print(f"Loaded {len(self.id2label)} labels")
                print(f"Label mapping: {self.id2label}")
            else:
                # Use config labels as fallback
                self.id2label = config.ID2LABEL
                self.label2id = config.LABEL2ID
                print("Using config labels as fallback")

            print(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            print("Using default model as fallback...")
            self._load_default_model()
    
    def _load_default_model(self):
        """Load default model as fallback - trained from scratch"""
        try:
            print("Loading base DistilBERT model...")
            num_labels = len(config.LABELS)
            self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                "distilbert-base-uncased",
                num_labels=num_labels,
                trust_remote_code=True
            )
            self.model.to(self.device)
            self.model.eval()
            self.id2label = config.ID2LABEL
            self.label2id = config.LABEL2ID
            print("[WARN] Using untrained base model - predictions will be random")
        except Exception as e:
            print(f"[ERROR] Failed to load default model: {e}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                config.MODEL_NAME,
                num_labels=len(config.LABELS),
                id2label=config.ID2LABEL,
                label2id=config.LABEL2ID
            )
            self.model.to(self.device)
            self.model.eval()
            
            self.id2label = config.ID2LABEL
            self.label2id = config.LABEL2ID
            
            print("Loaded default model as fallback")
        except Exception as e:
            print(f"Failed to load default model: {str(e)}")
            raise
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for classification"""
        if not text or len(text.strip()) < config.MIN_TEXT_LENGTH:
            return ""
        
        # Clean text
        text = ' '.join(text.split())
        text = text.replace('\n', ' ')
        text = text[:config.EXTRACT_FIRST_N_CHARS]
        
        return text.strip()
    
    def predict_single(self, text: str) -> Dict:
        """
        Predict class for a single text
        
        Args:
            text: Input text
        
        Returns:
            Dictionary with prediction results
        """
        if not text:
            return self._create_error_result("Empty text provided")
        
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        if len(processed_text) < config.MIN_TEXT_LENGTH:
            return self._create_error_result(f"Text too short (minimum {config.MIN_TEXT_LENGTH} chars)")
        
        try:
            # Tokenize
            inputs = self.tokenizer(
                processed_text,
                truncation=True,
                padding=True,
                max_length=config.MAX_LENGTH,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Predict with temperature scaling for better calibration
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                
                # Temperature scaling: T=1.5 smooths overconfident predictions
                temperature = 1.5
                scaled_logits = logits / temperature
                probabilities = torch.softmax(scaled_logits, dim=-1)
                predicted_id = torch.argmax(probabilities, dim=-1).item()

            # Get probabilities for all classes
            probs = probabilities[0].cpu().numpy()

            # Get label - id2label keys are strings from JSON
            if str(predicted_id) in self.id2label:
                predicted_label = self.id2label[str(predicted_id)]
            elif predicted_id in self.id2label:
                predicted_label = self.id2label[predicted_id]
            else:
                # Fallback to config labels
                if predicted_id < len(config.LABELS):
                    predicted_label = config.LABELS[predicted_id]
                else:
                    predicted_label = f"label_{predicted_id}"

            # Get top predictions
            top_indices = np.argsort(probs)[-3:][::-1]

            # Prepare all probabilities dictionary using string keys
            all_probs = {}
            for i, prob in enumerate(probs):
                label_key = str(i)
                if label_key in self.id2label:
                    label = self.id2label[label_key]
                elif i in self.id2label:
                    label = self.id2label[i]
                elif i < len(config.LABELS):
                    label = config.LABELS[i]
                else:
                    label = f"label_{i}"
                all_probs[label] = float(prob)

            # Prepare top predictions list using string keys
            top_predictions = []
            for idx in top_indices:
                label_key = str(idx)
                if label_key in self.id2label:
                    label = self.id2label[label_key]
                elif idx in self.id2label:
                    label = self.id2label[idx]
                elif idx < len(config.LABELS):
                    label = config.LABELS[idx]
                else:
                    label = f"label_{idx}"

                top_predictions.append({
                    'label': label,
                    'confidence': float(probs[idx]),
                    'label_id': idx
                })

            return {
                'predicted_label': predicted_label,
                'predicted_label_id': predicted_id,
                'confidence': float(probs[predicted_id]),
                'all_probabilities': all_probs,
                'top_predictions': top_predictions,
                'text_length': len(processed_text),
                'status': 'success'
            }

        except Exception as e:
            print(f"Prediction error: {str(e)}")
            import traceback
            traceback.print_exc()
            return self._create_error_result(f"Prediction failed: {str(e)}")
    
    def predict_from_pdf(self, pdf_path: str) -> Dict:
        """
        Predict class for a PDF file
        
        Args:
            pdf_path: Path to PDF file
        
        Returns:
            Dictionary with prediction results including file info
        """
        # Extract text from PDF
        extractor = PDFTextExtractor()
        text = extractor.extract_text(pdf_path)
        
        if not text:
            return self._create_error_result(f"Could not extract text from PDF: {pdf_path}")
        
        # Get prediction
        result = self.predict_single(text)
        
        # Add file information
        result['file_path'] = pdf_path
        result['file_name'] = Path(pdf_path).name
        
        return result
    
    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """
        Predict classes for a batch of texts
        
        Args:
            texts: List of input texts
        
        Returns:
            List of prediction results
        """
        results = []
        for i, text in enumerate(texts):
            if i % 10 == 0:
                print(f"Processing text {i+1}/{len(texts)}...")
            result = self.predict_single(text)
            result['text_index'] = i
            results.append(result)
        
        return results
    
    def predict_bulk_pdfs(self, pdf_paths: List[str]) -> List[Dict]:
        """
        Predict classes for multiple PDF files
        
        Args:
            pdf_paths: List of PDF file paths
        
        Returns:
            List of prediction results
        """
        results = []
        
        print(f"Processing {len(pdf_paths)} PDF files...")
        
        for i, pdf_path in enumerate(pdf_paths):
            if i % 5 == 0:
                print(f"Processing PDF {i+1}/{len(pdf_paths)}...")
            
            result = self.predict_from_pdf(pdf_path)
            results.append(result)
        
        return results
    
    def predict_csv(self, 
                   input_csv: str, 
                   text_column: str = 'text',
                   output_csv: Optional[str] = None) -> pd.DataFrame:
        """
        Predict classes for texts in a CSV file
        
        Args:
            input_csv: Path to input CSV file
            text_column: Name of column containing text
            output_csv: Path to save predictions (optional)
        
        Returns:
            DataFrame with predictions
        """
        # Read input CSV
        df = pd.read_csv(input_csv)
        
        print(f"Processing {len(df)} rows from {input_csv}")
        
        # Predict for each row
        predictions = self.predict_batch(df[text_column].astype(str).tolist())
        
        # Add predictions to DataFrame
        results_df = df.copy()
        results_df['predicted_label'] = [p['predicted_label'] for p in predictions]
        results_df['predicted_label_id'] = [p['predicted_label_id'] for p in predictions]
        results_df['confidence'] = [p['confidence'] for p in predictions]
        results_df['prediction_status'] = [p['status'] for p in predictions]
        
        # Save if output path provided
        if output_csv:
            results_df.to_csv(output_csv, index=False)
            print(f"Predictions saved to: {output_csv}")
        
        return results_df
    
    def export_bulk_predictions(self, 
                               pdf_paths: List[str], 
                               output_path: Optional[str] = None) -> pd.DataFrame:
        """
        Process bulk PDFs and export results
        
        Args:
            pdf_paths: List of PDF file paths
            output_path: Path to save results (optional)
        
        Returns:
            DataFrame with predictions
        """
        if output_path is None:
            output_path = config.BULK_PREDICTIONS_PATH
        
        # Get predictions
        predictions = self.predict_bulk_pdfs(pdf_paths)
        
        # Convert to DataFrame
        results_data = []
        for pred in predictions:
            row = {
                'file_path': pred.get('file_path', ''),
                'file_name': pred.get('file_name', ''),
                'predicted_label': pred.get('predicted_label', 'error'),
                'predicted_label_id': pred.get('predicted_label_id', -1),
                'confidence': pred.get('confidence', 0.0),
                'text_length': pred.get('text_length', 0),
                'top_alternative': '',
                'top_confidence': 0.0,
                'status': pred.get('status', 'error')
            }
            
            # Add top alternative prediction
            top_preds = pred.get('top_predictions', [])
            if len(top_preds) > 1:
                row['top_alternative'] = top_preds[1]['label']
                row['top_confidence'] = top_preds[1]['confidence']
            
            results_data.append(row)
        
        results_df = pd.DataFrame(results_data)
        
        # Save results
        results_df.to_csv(output_path, index=False)
        print(f"Bulk predictions exported to: {output_path}")
        
        # Print summary
        self._print_summary(results_df)
        
        return results_df
    
    def _print_summary(self, results_df: pd.DataFrame):
        """Print summary of predictions"""
        print("\n" + "="*60)
        print("PREDICTION SUMMARY")
        print("="*60)
        print(f"Total documents: {len(results_df)}")
        
        # Count by label
        if 'predicted_label' in results_df.columns:
            label_counts = results_df['predicted_label'].value_counts()
            print("\nPredictions by label:")
            for label, count in label_counts.items():
                print(f"  {label}: {count} documents")
        
        # Success rate
        success_count = len(results_df[results_df['status'] == 'success'])
        print(f"\nSuccess rate: {success_count}/{len(results_df)} ({success_count/len(results_df)*100:.1f}%)")
        
        # Average confidence
        if 'confidence' in results_df.columns:
            avg_conf = results_df['confidence'].mean()
            print(f"Average confidence: {avg_conf:.3f}")
        
        print("="*60)
    
    def _create_error_result(self, error_msg: str) -> Dict:
        """Create error result dictionary"""
        return {
            'predicted_label': 'error',
            'predicted_label_id': -1,
            'confidence': 0.0,
            'all_probabilities': {},
            'top_predictions': [],
            'error': error_msg,
            'status': 'error'
        }


def interactive_demo():
    """Interactive demo for document classification"""
    predictor = DocumentClassifierPredictor()
    
    print("="*60)
    print("DOCUMENT CLASSIFICATION DEMO")
    print("="*60)
    print(f"Supported labels: {', '.join(config.LABELS)}")
    print("Type 'quit' to exit")
    print("="*60)
    
    while True:
        try:
            # Get input
            print("\nEnter document text (or type 'pdf <path>' to classify a PDF file):")
            user_input = input("> ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            # Check if it's a PDF file
            if user_input.lower().startswith('pdf '):
                pdf_path = user_input[4:].strip()
                if Path(pdf_path).exists():
                    print(f"Classifying PDF: {pdf_path}")
                    result = predictor.predict_from_pdf(pdf_path)
                else:
                    print(f"File not found: {pdf_path}")
                    continue
            else:
                # Classify text
                result = predictor.predict_single(user_input)
            
            # Display results
            print("\n" + "-"*40)
            print("CLASSIFICATION RESULTS")
            print("-"*40)
            
            if result['status'] == 'success':
                print(f"Predicted label: {result['predicted_label']}")
                print(f"Confidence: {result['confidence']:.3f}")
                print(f"Text length: {result['text_length']} characters")
                
                print("\nTop predictions:")
                for i, pred in enumerate(result['top_predictions'][:3]):
                    prefix = "✓ " if i == 0 else "  "
                    print(f"{prefix}{pred['label']}: {pred['confidence']:.3f}")
                
                print("\nAll probabilities:")
                for label, prob in result['all_probabilities'].items():
                    print(f"  {label}: {prob:.3f}")
            else:
                print(f"Error: {result.get('error', 'Unknown error')}")
            
            print("-"*40)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {str(e)}")


def main():
    """Main function for bulk prediction"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Document Classifier Predictor")
    parser.add_argument("--mode", choices=["single", "bulk", "csv", "interactive"],
                       default="interactive", help="Prediction mode")
    parser.add_argument("--input", type=str, help="Input text, file, or directory")
    parser.add_argument("--output", type=str, help="Output file path")
    parser.add_argument("--text-column", type=str, default="text",
                       help="Column name for text in CSV mode")
    
    args = parser.parse_args()
    
    predictor = DocumentClassifierPredictor()
    
    if args.mode == "interactive":
        interactive_demo()
    
    elif args.mode == "single":
        if not args.input:
            print("Please provide input text with --input")
            return
        
        if args.input.endswith('.pdf'):
            result = predictor.predict_from_pdf(args.input)
        else:
            result = predictor.predict_single(args.input)
        
        print(json.dumps(result, indent=2, default=str))
    
    elif args.mode == "bulk":
        if not args.input:
            print("Please provide input directory with --input")
            return
        
        # Get all PDFs from directory
        input_path = Path(args.input)
        if input_path.is_dir():
            pdf_paths = list(input_path.glob("**/*.pdf"))
        else:
            # Assume it's a file with list of paths
            with open(args.input, 'r') as f:
                pdf_paths = [line.strip() for line in f if line.strip()]
        
        print(f"Found {len(pdf_paths)} PDF files")
        results_df = predictor.export_bulk_predictions(pdf_paths, args.output)
    
    elif args.mode == "csv":
        if not args.input:
            print("Please provide input CSV with --input")
            return
        
        results_df = predictor.predict_csv(args.input, args.text_column, args.output)
        print(f"Processed {len(results_df)} rows")


if __name__ == "__main__":
    main()