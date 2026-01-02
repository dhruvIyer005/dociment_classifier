# fix_checkpoint.py (save in project root)
import shutil
import json
from pathlib import Path

def fix_checkpoint():
    """Fix missing config.json in checkpoint directory"""
    # Path to your model
    model_dir = Path("models/distilbert_classifier")
    checkpoint_dir = model_dir / "checkpoint-49"
    
    print(f"Model directory: {model_dir}")
    print(f"Checkpoint directory: {checkpoint_dir}")
    
    # Check if directories exist
    if not model_dir.exists():
        print(f"❌ Model directory not found: {model_dir}")
        return
    
    if not checkpoint_dir.exists():
        print(f"❌ Checkpoint directory not found: {checkpoint_dir}")
        return
    
    print(f"\nFixing checkpoint: {checkpoint_dir.name}")
    
    # 1. Copy config.json if it exists in parent
    config_src = model_dir / "config.json"
    if config_src.exists():
        shutil.copy2(config_src, checkpoint_dir / "config.json")
        print("✓ Copied config.json")
    else:
        # Create a config.json
        config = {
            "_name_or_path": "distilbert-base-uncased",
            "architectures": ["DistilBertForSequenceClassification"],
            "attention_probs_dropout_prob": 0.1,
            "dim": 768,
            "dropout": 0.1,
            "hidden_dim": 3072,
            "initializer_range": 0.02,
            "max_position_embeddings": 512,
            "model_type": "distilbert",
            "n_heads": 12,
            "n_layers": 6,
            "pad_token_id": 0,
            "qa_dropout": 0.1,
            "seq_classif_dropout": 0.2,
            "sinusoidal_pos_embds": False,
            "tie_weights_": True,
            "transformers_version": "4.30.2",
            "vocab_size": 30522,
            "num_labels": 5,  # Your 5 classes
            "id2label": {"0": "ieee", "1": "springer", "2": "acm", "3": "compliance", "4": "legal"},
            "label2id": {"ieee": 0, "springer": 1, "acm": 2, "compliance": 3, "legal": 4}
        }
        
        with open(checkpoint_dir / "config.json", 'w') as f:
            json.dump(config, f, indent=2)
        print("✓ Created config.json")
    
    # 2. Copy other missing files
    files_to_copy = [
        "special_tokens_map.json",
        "tokenizer_config.json", 
        "tokenizer.json",
        "vocab.txt",
        "label_mapping.json"
    ]
    
    for file_name in files_to_copy:
        src = model_dir / file_name
        dst = checkpoint_dir / file_name
        
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
            print(f"✓ Copied {file_name}")
    
    # 3. Also ensure pytorch_model.bin exists in checkpoint
    if not (checkpoint_dir / "pytorch_model.bin").exists():
        # Check if training_args.bin can be renamed
        training_args = checkpoint_dir / "training_args.bin"
        if training_args.exists():
            # In some cases, training_args.bin is actually the model
            shutil.copy2(training_args, checkpoint_dir / "pytorch_model.bin")
            print("✓ Copied training_args.bin as pytorch_model.bin")
    
    print("\n✅ Checkpoint fixed!")
    print(f"\nFiles in checkpoint directory:")
    for file in checkpoint_dir.iterdir():
        print(f"  - {file.name}")

if __name__ == "__main__":
    fix_checkpoint()