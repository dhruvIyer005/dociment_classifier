# Document Classifier

Multi-document PDF classification system using Legal-BERT (nlpaueb/legal-bert-base-uncased).

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Add training data (20+ PDFs per folder)
data/ACM/
data/IEEE/
data/Springer/
data/Legal/
data/Compliance/

# 3. Train model
python train.py

# 4. Run web app
python app/flask_app.py
# Visit http://127.0.0.1:5000

# 5. Or use CLI
python predict.py doc1.pdf doc2.pdf
```

## Structure

```
data/               # Training PDFs
├── ACM/
├── IEEE/
├── Springer/
├── Legal/
└── Compliance/

models/             # Trained model weights
└── legal_bert_classifier/

app/                # Web application
├── flask_app.py
└── templates/
    └── index.html

src/                # Core modules
├── classifier.py
├── text_processor.py
└── config.py

```

## Configuration

Edit `src/config.py` to adjust:
- Batch size (default: 8)
- Learning rate (default: 2e-5)
- Epochs (default: 4)
- Chunk size (default: 425 tokens)

## Model

- **Base**: Legal-BERT (nlpaueb/legal-bert-base-uncased)
- **Task**: 5-class classification (ACM, IEEE, Springer, Legal, Compliance)
- **Method**: Chunk-based inference with logit averaging
- **GPU**: RTX 4050 optimized
- **Training time**: 30-60 minutes
- **Inference**: <2s per PDF
- **Expected accuracy**: 90-95%
- Flask 3.0+

See [web_app/requirements.txt](web_app/requirements.txt) for full dependency list.
