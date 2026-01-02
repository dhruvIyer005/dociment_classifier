import torch
from torch.utils.data import Dataset
from transformers import DistilBertTokenizer, Trainer
from config import MODEL_NAME, MAX_LENGTH, BASE_GRADING_CRITERIA, ENHANCED_TEMPLATE_TYPES
import torch.nn as nn
import numpy as np

class DocumentDataset(Dataset):
    def __init__(self, texts, template_labels, grading_scores, tokenizer):
        self.texts = texts
        self.template_labels = template_labels
        self.grading_scores = grading_scores
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=MAX_LENGTH,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'template_labels': torch.tensor(self.template_labels[idx], dtype=torch.long),
            'grading_labels': torch.tensor(self.grading_scores[idx], dtype=torch.float)
        }

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # Debug: ensure inputs contain expected keys
        if 'template_labels' not in inputs:
            try:
                print('compute_loss keys:', list(inputs.keys()))
            except Exception:
                pass

        template_logits, grading_outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask']
        )
        
        # Template classification loss
        template_loss = nn.CrossEntropyLoss()(template_logits, inputs['template_labels'])
        
        # Grading regression losses with proper scaling
        grading_loss = 0
        batch_size = inputs['grading_labels'].size(0)
        
        for i, grading_output in enumerate(grading_outputs):
            predicted_scores = grading_output.squeeze()
            actual_scores = inputs['grading_labels'][:, i]
            
            # Ensure predictions are in reasonable range (1-5)
            predicted_scores = torch.clamp(predicted_scores, 1.0, 5.0)
            grading_loss += nn.MSELoss()(predicted_scores, actual_scores)
        
        # Balance the losses - focus more on classification initially
        total_loss = template_loss + (grading_loss * 0.1)  # Reduced weight for grading
        
        return (total_loss, (template_logits, grading_outputs)) if return_outputs else total_loss

def create_enhanced_sample_data():
    """Create better sample data with clearer patterns and more diversity"""
    
    sample_data = {
        'text': [
            # HIGH QUALITY Research Papers (template_type: 3)
            "ABSTRACT: This paper introduces a novel transformer-based framework for document classification. We propose a multi-task learning approach that simultaneously identifies document types and assesses quality metrics. Our method achieves state-of-the-art performance on three benchmark datasets, demonstrating the effectiveness of combining structural and semantic features for comprehensive document understanding.",
            
            "INTRODUCTION: The exponential growth of digital documents necessitates automated classification systems. Traditional methods relying on handcrafted features struggle with document complexity and variability. In this work, we address these limitations by leveraging pre-trained language models fine-tuned on diverse document corpora. Our contributions include a hierarchical attention mechanism and multi-objective optimization for joint classification and quality assessment.",
            
            "METHODOLOGY: Our approach employs DistilBERT as the backbone architecture with custom classification and regression heads. For document type classification, we use a softmax layer over five categories. For quality assessment, we implement separate regression heads for four criteria, each trained to predict scores on a 1-5 scale. The model is optimized using a combined loss function with weighted components.",
            
            # MEDIUM QUALITY Research Papers
            "This study examines document classification methods. We use transformer models and evaluate on several datasets. Results show improved performance over baselines. The approach shows promise for practical applications.",
            
            "We present a method for classifying documents. The system uses neural networks and achieves good results. Future work will explore additional document types.",
            
            # HIGH QUALITY Compliance Documents (template_type: 1)
            "COMPLIANCE CERTIFICATION DOCUMENT - GDPR ARTICLE 35. ORGANIZATION: Data Solutions Inc. CERTIFICATION DATE: 2024-03-15. SCOPE: All data processing activities involving personal data of EU citizens. COMPLIANCE STATUS: FULLY COMPLIANT. ASSESSMENT: Comprehensive data protection impact assessment conducted. RISK MITIGATION: Encryption implemented, access controls established, breach response plan documented. VALIDATION: External audit completed by Certified GDPR Professionals LLC.",
            
            "HIPAA COMPLIANCE AUDIT REPORT. FACILITY: Metropolitan Healthcare Center. AUDIT DATE: 2024-02-20. SCOPE: Electronic Protected Health Information (ePHI) security controls. FINDINGS: All required administrative, physical, and technical safeguards implemented. DOCUMENTATION: Policies and procedures comprehensively documented. TRAINING: All staff completed annual HIPAA compliance training. RECOMMENDATIONS: Continue quarterly security awareness updates.",
            
            # MEDIUM QUALITY Compliance Documents
            "Compliance report for data protection. Requirements met according to regulations. Staff training completed. Documentation available for review.",
            
            # HIGH QUALITY Legal Documents (template_type: 4)
            "CONFIDENTIALITY AND NON-DISCLOSURE AGREEMENT. THIS AGREEMENT made and entered into as of March 15, 2024, by and between INNOVATIVE SOLUTIONS LLC, a Delaware limited liability company with its principal office at 123 Business Avenue, and TECH PARTNERS INC., a California corporation with its principal office at 456 Innovation Drive. RECITALS: WHEREAS, the parties wish to explore a business relationship; WHEREAS, disclosure of confidential information may be necessary.",
            
            "SOFTWARE LICENSE AGREEMENT. This Software License Agreement (the 'Agreement') is entered into between SOFTWARE PROVIDER CORP. ('Licensor') and LICENSEE COMPANY ('Licensee'). GRANT OF LICENSE: Subject to the terms of this Agreement, Licensor grants Licensee a non-exclusive, non-transferable license to use the Software. TERM: This Agreement shall commence on the Effective Date and continue for one year.",
            
            # MEDIUM QUALITY Legal Documents
            "Legal agreement between parties. Terms and conditions specified. Rights and obligations outlined. Effective date included.",
            
            # HIGH QUALITY Business Reports (template_type: 0)
            "QUARTERLY FINANCIAL REPORT - Q1 2024. EXECUTIVE SUMMARY: Company achieved record revenue of $15 million, representing 25% year-over-year growth. Key drivers include successful product launches and market expansion. FINANCIAL HIGHLIGHTS: Revenue: $15M (↑25%), Operating Income: $3.5M (↑30%), Net Profit: $2.8M (↑28%). OPERATIONAL METRICS: Customer acquisition: 15,000 new clients, Customer retention: 92%.",
            
            "ANNUAL STRATEGIC PLAN 2024-2025. VISION: To become the leading provider of AI-powered document solutions. MISSION: Deliver innovative, scalable document intelligence platforms. STRATEGIC OBJECTIVES: 1. Expand product portfolio with three new offerings 2. Enter two new geographic markets 3. Achieve 40% revenue growth 4. Establish industry thought leadership.",
            
            # MEDIUM QUALITY Business Reports
            "Quarterly report showing business performance. Revenue increased compared to last year. New customers acquired. Future plans outlined.",
            
            # HIGH QUALITY Proposals (template_type: 2)
            "PROJECT PROPOSAL: AI-Powered Document Processing System. CLIENT: Global Financial Services Inc. OBJECTIVES: Automate document classification for 50,000 monthly documents, reduce processing time by 70%, improve accuracy to 98%. SCOPE: System development, integration with existing infrastructure, staff training, ongoing support. BUDGET: $300,000 over 8 months. DELIVERABLES: Custom AI model, API integration, analytics dashboard, documentation.",
            
            "RESEARCH PROPOSAL: Advanced Document Understanding. FUNDING AGENCY: National Science Foundation. PROJECT DURATION: 24 months. BUDGET REQUEST: $500,000. RESEARCH QUESTIONS: 1. How can transformer models be optimized for multi-domain document classification? 2. What architectural improvements enhance quality assessment accuracy? METHODOLOGY: Comparative analysis of transformer architectures, ablation studies, large-scale evaluation.",
            
            # MEDIUM QUALITY Proposals
            "Project proposal for new system development. Objectives and timeline specified. Budget outlined. Expected benefits described.",
            
            # LOW QUALITY Examples (for contrast)
            "document about stuff. some information here. not very detailed.",
            "report with basic content. needs more work. incomplete sections.",
            "quick draft. ideas presented but not developed. requires revision.",
        ],
        'template_type': [
            0, 0, 0, 0, 0,  # ieee research_paper
            1, 1, 1,        # springer
            2, 2, 2,        # acm
            3, 3, 3,        # compliance
            4, 4, 4,        # legal
            0, 3, 4         # low quality examples
        ],
        'completeness': [5, 5, 5, 3, 2, 5, 5, 3, 5, 5, 3, 5, 5, 3, 5, 5, 3, 1, 1, 2],
        'clarity':      [5, 5, 4, 3, 2, 5, 5, 3, 5, 4, 3, 5, 5, 3, 5, 5, 3, 1, 2, 2],
        'structure':    [5, 5, 5, 2, 2, 5, 5, 3, 5, 5, 3, 5, 5, 3, 5, 5, 3, 1, 1, 1],
        'professionalism': [5, 5, 5, 3, 2, 5, 5, 3, 5, 5, 3, 5, 5, 3, 5, 5, 3, 1, 1, 1]
    }
    return sample_data