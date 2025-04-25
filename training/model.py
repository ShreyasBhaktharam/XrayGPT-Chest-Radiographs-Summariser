# training/model.py
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
import logging

logger = logging.getLogger(__name__)

class XrayGPTModel(nn.Module):
    """
    Model for combining X-ray image features with text for report generation
    """
    def __init__(self, model_type="bio-gpt", num_classes=14):
        super(XrayGPTModel, self).__init__()
        
        # Initialize pretrained language model
        self.model_type = model_type
        if model_type == "bio-gpt":
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/biogpt")
            self.text_model = AutoModel.from_pretrained("microsoft/biogpt")
        else:
            # Default to biogpt
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/biogpt")
            self.text_model = AutoModel.from_pretrained("microsoft/biogpt")
        
        # Image feature extractor (placeholder - would use CNN in full implementation)
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3),  # Changed input channels from 1 to 3 for RGB
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 512),
            nn.ReLU()
        )
        
        # Metadata embedding
        self.gender_embedding = nn.Embedding(2, 16)  # M/F
        self.view_position_embedding = nn.Embedding(10, 32)  # Different view positions
        self.age_embedding = nn.Linear(1, 16)  # Age as continuous feature
        
        # Multimodal fusion
        metadata_dim = 16 + 32 + 16  # Combined metadata dimensions
        self.fusion = nn.Sequential(
            nn.Linear(512 + self.text_model.config.hidden_size + metadata_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )
        
        # View position mapping
        self.view_position_map = {
            'PA': 0, 'AP': 1, 'LATERAL': 2, 'L': 3, 'R': 4, 
            'AP SUPINE': 5, 'PORTABLE': 6, 'AP ERECT': 7, 'LL': 8, 'RL': 9
        }
        
        # Gender mapping
        self.gender_map = {'M': 0, 'F': 1}

    def forward(self, images, input_ids, attention_mask, metadata=None):
        # Process metadata if provided
        metadata_features = None
        if metadata:
            # Gender embedding
            gender_indices = torch.tensor([self.gender_map.get(g, 0) for g in metadata['patient_gender']], 
                                         device=images.device)
            gender_embedding = self.gender_embedding(gender_indices)
            
            # View position embedding
            view_indices = torch.tensor([self.view_position_map.get(v, 0) for v in metadata['view_position']], 
                                        device=images.device)
            view_embedding = self.view_position_embedding(view_indices)
            
            # Age embedding (normalize age)
            age_values = torch.tensor([[float(a)] for a in metadata['patient_age']], 
                                     device=images.device).float() / 100.0
            age_embedding = self.age_embedding(age_values)
            
            # Combine metadata features
            metadata_features = torch.cat([gender_embedding, view_embedding, age_embedding], dim=1)
        
        # Image features
        image_features = self.image_encoder(images)
        
        # Text features
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        text_features = text_outputs.last_hidden_state[:, 0, :]
        
        # Concatenate features
        if metadata_features is not None:
            combined_features = torch.cat([image_features, text_features, metadata_features], dim=1)
        else:
            combined_features = torch.cat([image_features, text_features], dim=1)
        
        # Classification through fusion layer
        outputs = self.fusion(combined_features)
        return outputs

    def save_pretrained(self, path):
        """Save model to disk"""
        torch.save({
            'model_type': self.model_type,
            'state_dict': self.state_dict(),
        }, path)
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def from_pretrained(cls, path):
        """Load model from disk"""
        checkpoint = torch.load(path)
        model = cls(model_type=checkpoint['model_type'])
        model.load_state_dict(checkpoint['state_dict'])
        logger.info(f"Model loaded from {path}")
        return model