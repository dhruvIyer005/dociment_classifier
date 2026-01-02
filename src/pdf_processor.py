# pdf_processor.py
import os
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import fitz  # PyMuPDF
import PyPDF2
from config import config


class PDFTextExtractor:
    """Extract text from PDF files for document classification"""
    
    def __init__(self):
        self.min_text_length = config.MIN_TEXT_LENGTH
        self.extract_n_chars = config.EXTRACT_FIRST_N_CHARS
        
    def extract_text(self, pdf_path: str) -> Optional[str]:
        """
        Extract text from PDF file.
        Returns first N characters for classification.
        """
        try:
            # Try PyMuPDF first (faster and better)
            text = self._extract_with_pymupdf(pdf_path)
            if text:
                return self._preprocess_text(text)
            
            # Fallback to PyPDF2
            text = self._extract_with_pypdf2(pdf_path)
            if text:
                return self._preprocess_text(text)
            
            print(f"Error: Could not extract text from {pdf_path}")
            return None
            
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {str(e)}")
            return None
    
    def _extract_with_pymupdf(self, pdf_path: str) -> Optional[str]:
        """Extract text using PyMuPDF (fitz)"""
        try:
            doc = fitz.open(pdf_path)
            text = ""
            
            # Extract text from first few pages only (for classification)
            max_pages = min(10, len(doc))  # Don't need all pages for classification
            
            for page_num in range(max_pages):
                page = doc[page_num]
                page_text = page.get_text()
                text += page_text + "\n"
                
                # Stop if we have enough text
                if len(text) >= self.extract_n_chars:
                    break
            
            doc.close()
            return text.strip()
            
        except Exception as e:
            print(f"PyMuPDF extraction failed for {pdf_path}: {str(e)}")
            return None
    
    def _extract_with_pypdf2(self, pdf_path: str) -> Optional[str]:
        """Extract text using PyPDF2 (fallback)"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                max_pages = min(10, len(pdf_reader.pages))
                
                for page_num in range(max_pages):
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                    
                    # Stop if we have enough text
                    if len(text) >= self.extract_n_chars:
                        break
                
                return text.strip()
                
        except Exception as e:
            print(f"PyPDF2 extraction failed for {pdf_path}: {str(e)}")
            return None
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess extracted text for classification"""
        if not text or len(text.strip()) < self.min_text_length:
            return ""
        
        # Clean text
        text = ' '.join(text.split())  # Remove extra whitespace
        text = text.replace('\n', ' ')  # Replace newlines with spaces
        text = text[:self.extract_n_chars]  # Limit length
        
        return text.strip()
    
    def extract_text_with_metadata(self, pdf_path: str) -> Dict:
        """Extract text and basic metadata"""
        text = self.extract_text(pdf_path)
        
        if not text:
            return {}
        
        # Get basic metadata
        try:
            with fitz.open(pdf_path) as doc:
                page_count = len(doc)
        except:
            page_count = 0
        
        return {
            'text': text,
            'page_count': page_count,
            'file_path': pdf_path,
            'file_name': os.path.basename(pdf_path)
        }


class PDFDatasetBuilder:
    """Build dataset from PDF files for classification training"""
    
    def __init__(self):
        self.extractor = PDFTextExtractor()
        
    def process_pdfs_by_label(self, base_dir: str) -> pd.DataFrame:
        """
        Process PDFs organized by label directories:
        base_dir/
          ├── ieee/*.pdf
          ├── springer/*.pdf
          ├── acm/*.pdf
          ├── compliance/*.pdf
          └── legal/*.pdf
        """
        data = []
        base_path = Path(base_dir)
        
        print(f"Processing PDFs from: {base_dir}")
        
        for label in config.LABELS:
            label_dir = base_path / label
            
            if not label_dir.exists():
                print(f"Warning: Label directory '{label}' not found in {base_dir}")
                continue
            
            print(f"Processing {label} documents...")
            pdf_files = list(label_dir.glob("*.pdf"))
            
            if not pdf_files:
                print(f"No PDFs found in {label_dir}")
                continue
            
            for pdf_file in pdf_files:
                print(f"  - {pdf_file.name}")
                
                # Extract text
                result = self.extractor.extract_text_with_metadata(str(pdf_file))
                
                if result and result.get('text'):
                    # Create data entry
                    entry = {
                        'text': result['text'],
                        'label': label,
                        'label_id': config.LABEL2ID[label],
                        'file_name': result['file_name'],
                        'file_path': str(pdf_file),
                        'page_count': result['page_count']
                    }
                    data.append(entry)
                    print(f"    Extracted {len(result['text'])} characters")
                else:
                    print(f"    Failed to extract text")
        
        # Create DataFrame
        if data:
            df = pd.DataFrame(data)
            print(f"Successfully processed {len(df)} PDFs")
            return df
        else:
            print("No PDFs were successfully processed")
            return pd.DataFrame()
    
    def process_single_pdf(self, pdf_path: str, label: Optional[str] = None) -> Dict:
        """Process a single PDF file for prediction"""
        result = self.extractor.extract_text_with_metadata(pdf_path)
        
        if result and result.get('text'):
            return {
                'text': result['text'],
                'label': label,
                'file_name': result['file_name'],
                'file_path': pdf_path,
                'page_count': result['page_count']
            }
        return {}
    
    def process_bulk_pdfs(self, pdf_paths: List[str]) -> pd.DataFrame:
        """Process multiple PDFs for bulk classification"""
        data = []
        
        print(f"Processing {len(pdf_paths)} PDFs for bulk classification...")
        
        for pdf_path in pdf_paths:
            result = self.extractor.extract_text_with_metadata(pdf_path)
            
            if result and result.get('text'):
                entry = {
                    'text': result['text'],
                    'file_name': result['file_name'],
                    'file_path': pdf_path,
                    'page_count': result['page_count'],
                    'label': None,  # To be predicted
                    'label_id': None
                }
                data.append(entry)
                print(f"  Processed: {result['file_name']}")
            else:
                print(f"  Failed: {os.path.basename(pdf_path)}")
        
        return pd.DataFrame(data)


# For backward compatibility
class PDFProcessor(PDFDatasetBuilder):
    """Legacy class name support"""
    pass


def extract_text_for_classification(pdf_path: str) -> str:
    """Simple function to extract text from PDF for classification"""
    extractor = PDFTextExtractor()
    return extractor.extract_text(pdf_path) or ""