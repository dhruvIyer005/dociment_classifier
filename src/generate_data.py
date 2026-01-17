# generate_data.py
import pandas as pd
import numpy as np
import random
from pathlib import Path
from typing import List, Dict, Optional
from config import config
import json


class SyntheticDataGenerator:
    """Generate synthetic training data for document classification"""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        
        # Load real examples if available for better generation
        self.real_examples = self._load_real_examples()
        
        # Template patterns for each document type
        self.templates = self._initialize_templates()
        
    def _load_real_examples(self) -> Optional[pd.DataFrame]:
        """Load real PDF examples to inform synthetic generation"""
        real_data_path = config.RAW_PDFS_DIR
        if real_data_path.exists():
            # This would be populated after processing real PDFs
            return None
        return None
    
    def _initialize_templates(self) -> Dict:
        """Initialize document templates for each label"""
        return {
            "ieee": [
                "IEEE TRANSACTIONS ON {FIELD}\n\n{CONTENT}\n\nREFERENCES\n[1] Author, \"Title,\" Journal, vol., no., pp., year.",
                "Proceedings of the IEEE {CONFERENCE}\n\n{CONTENT}\n\nACKNOWLEDGMENT\nThis work was supported by {FUNDING}.",
                "IEEE Standard {NUMBER}\n\n{CONTENT}\n\nAPPENDIX\nSupplementary material is available."
            ],
            "springer": [
                "{TITLE}\n\nAbstract\n{CONTENT}\n\nKeywords: {KEYWORDS}\n\n1 Introduction\n{CONTENT}",
                "Springer {JOURNAL}\n\n{CONTENT}\n\nReceived: {DATE} / Accepted: {DATE} / Published: {DATE}",
                "Lecture Notes in {FIELD}\n\n{CONTENT}\n\n© Springer-Verlag {YEAR}"
            ],
            "acm": [
                "ACM {CONFERENCE}\n\n{CONTENT}\n\nCCS Concepts: {CONCEPTS}\n\nAdditional Key Words and Phrases: {KEYWORDS}",
                "{TITLE}\n\nABSTRACT\n{CONTENT}\n\n1. INTRODUCTION\n{CONTENT}\n\nREFERENCES",
                "ACM Transactions on {FIELD}\n\n{CONTENT}\n\nPermission to make digital or hard copies of part or all of this work for personal or classroom use is granted without fee."
            ],
            "compliance": [
                "COMPLIANCE POLICY DOCUMENT\n\n{CONTENT}\n\nEffective Date: {DATE}\nReview Date: {DATE}\n\nApproved by: {AUTHORITY}",
                "REGULATORY COMPLIANCE CHECKLIST\n\n{CONTENT}\n\nStatus: Compliant/Non-Compliant\nAction Required: {ACTIONS}",
                "DATA PROTECTION AND PRIVACY POLICY\n\n{CONTENT}\n\nThis policy is in accordance with {REGULATION} regulations."
            ],
            "legal": [
                "CONFIDENTIALITY AGREEMENT\n\n{CONTENT}\n\nIN WITNESS WHEREOF, the parties have executed this Agreement as of the Effective Date.",
                "LEGAL OPINION\n\n{CONTENT}\n\nBased on the foregoing analysis, it is our opinion that {CONCLUSION}.",
                "TERMS AND CONDITIONS\n\n{CONTENT}\n\nGoverning Law: This Agreement shall be governed by the laws of {JURISDICTION}."
            ]
        }
    
    def _generate_ieee_content(self) -> str:
        """Generate IEEE-style content"""
        fields = ["Signal Processing", "Computer Science", "Electrical Engineering", 
                 "Biomedical Engineering", "Communications"]
        topics = [
            "novel algorithm for signal denoising",
            "machine learning approach for image classification",
            "hardware implementation of neural networks",
            "optimization of wireless communication protocols",
            "deep learning models for medical diagnosis"
        ]
        
        field = random.choice(fields)
        topic = random.choice(topics)
        
        content_parts = [
            f"This paper presents a {topic}. The proposed method achieves significant improvements over existing approaches.",
            f"Experimental results demonstrate a {random.randint(10, 50)}% improvement in performance metrics.",
            f"The methodology section details the experimental setup and evaluation criteria.",
            f"Future work will explore extensions to real-world applications."
        ]
        
        return f"This work addresses challenges in {field} through a {topic}. " + " ".join(random.sample(content_parts, 2))
    
    def _generate_springer_content(self) -> str:
        """Generate Springer-style content"""
        fields = ["Computer Science", "Physics", "Mathematics", "Biology", "Chemistry"]
        topics = [
            "computational analysis of complex systems",
            "theoretical framework for modeling phenomena",
            "experimental validation of hypotheses",
            "statistical analysis of large datasets",
            "simulation of biological processes"
        ]
        
        field = random.choice(fields)
        topic = random.choice(topics)
        
        content_parts = [
            f"This study investigates {topic} within the context of {field}.",
            f"Methodological considerations are discussed in detail.",
            f"Results indicate statistically significant findings (p < 0.0{random.randint(1,5)}).",
            f"The discussion section interprets these findings in light of current literature.",
            f"Conclusions highlight implications for future research directions."
        ]
        
        return " ".join(random.sample(content_parts, 3))
    
    def _generate_acm_content(self) -> str:
        """Generate ACM-style content"""
        areas = ["Human-Computer Interaction", "Artificial Intelligence", 
                "Software Engineering", "Data Science", "Cybersecurity"]
        topics = [
            "user interface design evaluation",
            "deep learning model optimization",
            "software testing methodologies",
            "big data processing frameworks",
            "security vulnerability assessment"
        ]
        
        area = random.choice(areas)
        topic = random.choice(topics)
        
        content_parts = [
            f"This research contributes to the field of {area} by exploring {topic}.",
            f"A user study with {random.randint(20, 200)} participants was conducted.",
            f"Quantitative and qualitative measures were employed for evaluation.",
            f"The system architecture and implementation details are provided.",
            f"Limitations and potential improvements are discussed."
        ]
        
        return " ".join(random.sample(content_parts, 3))
    
    def _generate_compliance_content(self) -> str:
        """Generate compliance document content"""
        regulations = ["GDPR", "HIPAA", "SOX", "PCI DSS", "ISO 27001"]
        aspects = ["data protection", "access control", "audit trails", 
                  "risk assessment", "incident response"]
        
        regulation = random.choice(regulations)
        aspect = random.choice(aspects)
        
        content_parts = [
            f"This document outlines procedures for {aspect} in compliance with {regulation}.",
            f"All employees must complete annual training on these protocols.",
            f"Regular audits will be conducted to ensure ongoing compliance.",
            f"Violations of these policies may result in disciplinary action.",
            f"Updates to this document will be reviewed {random.choice(['quarterly', 'biannually', 'annually'])}."
        ]
        
        return " ".join(random.sample(content_parts, 3))
    
    def _generate_legal_content(self) -> str:
        """Generate legal document content"""
        doc_types = ["agreement", "contract", "memorandum", "opinion", "disclosure"]
        parties = ["Company", "Individual", "Partnership", "Corporation", "Entity"]
        
        doc_type = random.choice(doc_types)
        party_a = random.choice(parties)
        party_b = random.choice([p for p in parties if p != party_a])
        
        content_parts = [
            f"This {doc_type} is entered into between {party_a} and {party_b}.",
            f"The terms and conditions herein are legally binding and enforceable.",
            f"Any disputes shall be resolved through {random.choice(['arbitration', 'mediation', 'litigation'])}.",
            f"Confidential information shall not be disclosed to third parties.",
            f"This document shall be governed by the laws of the State of {random.choice(['California', 'New York', 'Texas', 'Delaware'])}."
        ]
        
        return " ".join(random.sample(content_parts, 3))
    
    def _fill_template(self, template: str, label: str) -> str:
        """Fill template with generated content"""
        content_generators = {
            "ieee": self._generate_ieee_content,
            "springer": self._generate_springer_content,
            "acm": self._generate_acm_content,
            "compliance": self._generate_compliance_content,
            "legal": self._generate_legal_content
        }
        
        # Generate content
        content = content_generators[label]()
        
        # Fill placeholders
        replacements = {
            "{FIELD}": random.choice(["Computer Science", "Engineering", "Technology", "Science"]),
            "{CONTENT}": content,
            "{CONFERENCE}": random.choice(["Conference", "Symposium", "Workshop"]),
            "{FUNDING}": random.choice(["NSF", "NIH", "DARPA", "Corporate Sponsorship"]),
            "{TITLE}": f"Research on {random.choice(['Machine Learning', 'Data Analysis', 'System Design'])}",
            "{KEYWORDS}": random.choice(["AI, ML, Deep Learning", "Security, Privacy, Compliance", "Algorithms, Optimization"]),
            "{JOURNAL}": random.choice(["Journal", "Transactions", "Letters"]),
            "{DATE}": f"{random.randint(1,12)}/{random.randint(1,28)}/{random.randint(2020,2024)}",
            "{YEAR}": str(random.randint(2020, 2024)),
            "{CONCEPTS}": random.choice(["• Computing methodologies → Machine learning", "• Security and privacy → Access control"]),
            "{AUTHORITY}": random.choice(["Compliance Committee", "Board of Directors", "CEO"]),
            "{REGULATION}": random.choice(["GDPR", "CCPA", "HIPAA"]),
            "{ACTIONS}": random.choice(["Immediate remediation required", "Monitoring recommended"]),
            "{CONCLUSION}": random.choice(["the proposed action is permissible", "further review is necessary"]),
            "{JURISDICTION}": random.choice(["California", "New York", "Federal"]),
            "{NUMBER}": f"{random.randint(100, 999)}.{random.randint(1, 9)}"
        }
        
        filled = template
        for placeholder, value in replacements.items():
            filled = filled.replace(placeholder, value)
        
        return filled
    
    def generate_examples(self, 
                         examples_per_class: int = 15,  # ~75 total for 5 classes
                         save_path: Optional[Path] = None) -> pd.DataFrame:
        """
        Generate synthetic training examples
        
        Args:
            examples_per_class: Number of examples per label
            save_path: Path to save generated data
        
        Returns:
            DataFrame with generated examples
        """
        if save_path is None:
            save_path = config.SYNTHETIC_DATA_DIR / "synthetic_training_data.csv"
        
        generated_data = []
        
        print(f"Generating {examples_per_class} examples per class...")
        
        for label in config.LABELS:
            print(f"  Generating {label} examples...")
            
            for i in range(examples_per_class):
                # Select template
                template = random.choice(self.templates[label])
                
                # Fill template
                text = self._fill_template(template, label)
                
                # Create example
                example = {
                    'text': text,
                    'label': label,
                    'label_id': config.LABEL2ID[label],
                    'source': 'synthetic',
                    'example_id': f"{label}_{i:03d}"
                }
                
                generated_data.append(example)
        
        # Create DataFrame
        df = pd.DataFrame(generated_data)
        
        # Save to CSV
        df.to_csv(save_path, index=False)
        
        print(f"Generated {len(df)} synthetic examples")
        print(f"Saved to: {save_path}")
        
        return df
    
    def augment_real_data(self, real_df: pd.DataFrame, 
                         augmentation_factor: int = 2) -> pd.DataFrame:
        """
        Augment real data with variations
        """
        augmented_data = []
        
        for _, row in real_df.iterrows():
            original_text = row['text']
            label = row['label']
            
            # Create variations
            for i in range(augmentation_factor):
                # Simple augmentation: add noise or modify slightly
                words = original_text.split()
                if len(words) > 50:
                    # Randomly replace some words
                    n_replace = max(1, len(words) // 100)
                    for _ in range(n_replace):
                        idx = random.randint(0, len(words)-1)
                        words[idx] = random.choice(['Additionally,', 'Furthermore,', 'Moreover,', 'Specifically,'])
                    
                    augmented_text = ' '.join(words)
                    
                    augmented_data.append({
                        'text': augmented_text,
                        'label': label,
                        'label_id': config.LABEL2ID[label],
                        'source': 'augmented',
                        'example_id': f"aug_{label}_{i}"
                    })
        
        return pd.DataFrame(augmented_data)


def generate_and_combine_datasets():
    """Main function to generate synthetic data and combine with real data"""
    generator = SyntheticDataGenerator()
    
    # Generate MORE synthetic data for better training
    synthetic_df = generator.generate_examples(examples_per_class=50)
    
    # Process real PDFs (if available)
    from pdf_processor import PDFDatasetBuilder
    
    builder = PDFDatasetBuilder()
    real_df = builder.process_pdfs_by_label(str(config.RAW_PDFS_DIR))
    
    if not real_df.empty:
        print(f"Loaded {len(real_df)} real examples")
        
        # Augment real data MORE aggressively
        augmented_df = generator.augment_real_data(real_df, augmentation_factor=4)
        
        # Combine all data
        combined_df = pd.concat([real_df, synthetic_df, augmented_df], ignore_index=True)
        
        # Shuffle
        combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Save combined dataset
        combined_path = config.DATA_DIR / "combined_training_data.csv"
        combined_df.to_csv(combined_path, index=False)
        
        print(f"Combined dataset: {len(combined_df)} examples")
        print(f"  - Real: {len(real_df)}")
        print(f"  - Synthetic: {len(synthetic_df)}")
        print(f"  - Augmented: {len(augmented_df)}")
        print(f"Saved to: {combined_path}")
        
        return combined_df
    else:
        print("No real data found, returning synthetic data only")
        return synthetic_df


if __name__ == "__main__":
    generate_and_combine_datasets()