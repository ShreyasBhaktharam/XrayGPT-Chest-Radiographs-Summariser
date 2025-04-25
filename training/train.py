# training/train.py
import os
import torch
from torch.utils.data import DataLoader, Dataset
import mlflow
import logging
import pandas as pd
from PIL import Image
from config.settings import MODELS_DIR, MLFLOW_TRACKING_URI, EXPERIMENT_NAME
from training.model import XrayGPTModel
from training.evaluate import evaluate_model
from torchvision import transforms

logger = logging.getLogger(__name__)

class XrayReportDataset(Dataset):
    """Dataset for X-ray images and reports"""
    def __init__(self, data_dir, tokenizer, csv_file, max_length=512):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.metadata = pd.read_csv(os.path.join(data_dir, csv_file))
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Get image path
        img_path = os.path.join(self.data_dir, self.metadata.iloc[idx]['path'])
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
            #convert image to tensor
            image = self.transform(image)
            #if self.transform:
                #image = self.transform(image)
                
            # Get report text
            report = self.metadata.iloc[idx]['finding']
            
            # Additional metadata (if needed)
            metadata = {
                'image_id': self.metadata.iloc[idx]['image_id'],
                'report_id': self.metadata.iloc[idx]['report_id'],
                'view_position': self.metadata.iloc[idx]['view_position'],
                'patient_gender': self.metadata.iloc[idx]['patient_gender'],
                'patient_age': self.metadata.iloc[idx]['patient_age']
            }
            
            # Here you would tokenize the report based on your model requirements
            # For example:
            # tokenized_report = self.tokenizer(report, padding='max_length', 
            #                                 max_length=512, truncation=True,
            #                                 return_tensors='pt')
            encoding = self.tokenizer(
            report,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
            # For now, just return the text
            return {
                'image': image,
                'report': report,
                'metadata': metadata
            }
            
        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            # Return a placeholder or skip
            # For simplicity, returning zeros
            return {
                'image': torch.zeros((3, 224, 224)),
                'report': "",
                'metadata': {}
            }
        
    '''
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Tokenize report text
        encoding = self.tokenizer(
            item['report'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Get image as tensor
        image = torch.tensor(item['image'], dtype=torch.float32).unsqueeze(0)  # Add channel dim
        
        return {
            'image': image,
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(item['labels'], dtype=torch.float32)
        }
    '''

def train_model(
    train_data_dir,
    val_data_dir,
    model_type="bio-gpt",
    batch_size=16,
    epochs=3,
    learning_rate=2e-5,
    save_dir=MODELS_DIR
):
    """Train the XrayGPT model"""
    # Initialize MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    # Create model and datasets
    model = XrayGPTModel(model_type=model_type)
    
    # Create dataset
    train_dataset = XrayReportDataset(train_data_dir, model.tokenizer, 'metadata.csv')
    val_dataset = XrayReportDataset(val_data_dir, model.tokenizer, 'metadata.csv')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    #criterion = torch.nn.BCEWithLogitsLoss()
    criterion = torch.nn.CrossEntropyLoss()
    
    # Training loop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    with mlflow.start_run() as run:
        # Log parameters
        mlflow.log_params({
            "model_type": model_type,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "epochs": epochs
        })
        
        for epoch in range(epochs):
            model.train()
            train_loss = 0
            
            for batch in train_loader:
                # Move batch to device
                images = batch['image'].to(device)
                
                # Process report text using tokenizer
                reports = batch['report']
                tokenized = model.tokenizer(
                    reports, 
                    padding='max_length', 
                    max_length=512, 
                    truncation=True,
                    return_tensors='pt'
                )
                
                input_ids = tokenized['input_ids'].to(device)
                attention_mask = tokenized['attention_mask'].to(device)
                
                # Assume labels are derived from reports
                # This would depend on your specific task
                # For example, for a classification task:
                # labels could be generated from the reports or other metadata
                
                # For this example, we'll assume the model output is compared to some target
                # You'll need to adjust this based on your actual model architecture and task
                labels = torch.ones(images.size(0), dtype=torch.long).to(device)  # Placeholder
                
                # Check if metadata fields exist in the batch
                # If not, we'll use empty placeholders
                metadata = {}
                
                # Check if metadata fields are directly in the batch
                if 'image_id' in batch:
                    print('IMAGE ID IN BATCH:', batch['image_id'])
                    metadata = {
                        'image_id': batch['image_id'],
                        'report_id': batch['report_id'],
                        'view_position': batch['view_position'],
                        'patient_gender': batch['patient_gender'],
                        'patient_age': batch['patient_age']
                    }
                # Check if metadata is nested in a 'metadata' key
                elif 'metadata' in batch:
                    print('METADATA IN BATCH:', batch['metadata'])
                    metadata = batch['metadata']
                # If neither format exists, create empty metadata
                else:
                    print('NO METADATA IN BATCH')
                    logger.warning("No metadata found in batch. Using placeholder values.")
                    batch_size = images.size(0)
                    metadata = {
                        'image_id': ['unknown'] * batch_size,
                        'report_id': ['unknown'] * batch_size,
                        'view_position': ['PA'] * batch_size,  # Default to PA view
                        'patient_gender': ['M'] * batch_size,  # Default to male
                        'patient_age': [50] * batch_size       # Default to age 50
                    }
                
                # Forward pass
                outputs = model(
                    images=images,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    metadata=metadata
                )
                
                loss = criterion(outputs, labels)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            
            # Evaluate on validation set
            val_metrics = evaluate_model(model, val_loader, criterion, device)
            
            # Log metrics
            mlflow.log_metrics({
                "train_loss": avg_train_loss,
                "val_loss": val_metrics['loss'],
                "val_f1": val_metrics['f1'],
                "val_precision": val_metrics['precision'],
                "val_recall": val_metrics['recall']
            }, step=epoch)
            
            logger.info(f"Epoch {epoch+1}/{epochs} - "
                       f"Train Loss: {avg_train_loss:.4f}, "
                       f"Val Loss: {val_metrics['loss']:.4f}, "
                       f"Val F1: {val_metrics['f1']:.4f}")
        
        # Save model
        os.makedirs(save_dir, exist_ok=True)
        model_path = os.path.join(save_dir, f"xraygpt_{model_type}.pt")

        #Commenting this out since I don't want my laptop to run out of memory
        model.save_pretrained(model_path)
        
        # Log model to MLflow
        mlflow.log_artifact(model_path)
    
    return model, run.info.run_id