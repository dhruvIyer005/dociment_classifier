"""
predict.py - Simple batch prediction script
Usage: python predict.py pdf1.pdf pdf2.pdf pdf3.pdf
"""

import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from classifier import create_classifier
from config import OUTPUT_DIR


def main():
    """Predict multiple PDFs"""
    
    if len(sys.argv) < 2:
        print("Usage: python predict.py <pdf1> <pdf2> ... <pdfN>")
        print("\nExample: python predict.py doc1.pdf doc2.pdf")
        sys.exit(1)
    
    pdf_files = sys.argv[1:]
    
    print("="*60)
    print("Legal-BERT Document Classifier - Batch Prediction")
    print("="*60)
    print()
    
    # Create classifier
    print("Loading model...")
    classifier = create_classifier()
    print("✓ Model loaded\n")
    
    # Predict
    results = classifier.batch_predict(pdf_files)
    
    # Display results
    print("Results:")
    print("-"*60)
    print(f"{'Filename':<30} {'Label':<15} {'Confidence':<10}")
    print("-"*60)
    
    for result in results:
        filename = result["filename"][:28]
        label = result["label"]
        conf = f"{result['confidence']:.1%}" if isinstance(result['confidence'], float) else result['confidence']
        print(f"{filename:<30} {label:<15} {conf:<10}")
    
    print("-"*60)
    print(f"Processed: {len(results)} files")
    
    # Save results
    output_file = OUTPUT_DIR / "predictions.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
