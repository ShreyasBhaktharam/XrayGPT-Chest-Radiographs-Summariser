# xraygpt_hf_integration.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download, login
import os
from PIL import Image
from torchvision import transforms
import json

class XrayGPTHuggingFace:
    def __init__(self, model_path="khanhduong/xraygpt_vicuna"):
        """Initialize XrayGPT from Hugging Face"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        login(token="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")
        
        # Download the model from HuggingFace
        print(f"Downloading model from {model_path}...")
        model_dir = snapshot_download(repo_id=model_path)
        
        # Load the model
        self.load_model(model_dir)
        
    def load_model(self, model_dir):
        """Load XrayGPT model components"""
        # Load configuration
        config_path = os.path.join(model_dir, "xraygpt_config.json")
        
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Load components (vision encoder, LLM, projection layer)
        self.load_vision_model(model_dir)
        self.load_language_model(model_dir)
        self.load_projection_layer(model_dir)
        
    def load_vision_model(self, model_dir):
        """Load MedClip vision encoder"""
        from transformers import CLIPModel, CLIPProcessor
        
        self.vision_model = CLIPModel.from_pretrained("flaviagiammarino/pubmed-clip-vit-base-patch32")
        self.vision_processor = CLIPProcessor.from_pretrained("flaviagiammarino/pubmed-clip-vit-base-patch32")
        
        # Load pretrained weights if available
        vision_weights_path = os.path.join(model_dir, "vision_model.pth")
        if os.path.exists(vision_weights_path):
            state_dict = torch.load(vision_weights_path, map_location=self.device)
            self.vision_model.load_state_dict(state_dict)
        
        self.vision_model.to(self.device)
        
    def load_language_model(self, model_dir):
        """Load Vicuna LLM"""
        # Note: You may need to use the actual vicuna model path
        self.tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5")
        self.language_model = AutoModelForCausalLM.from_pretrained("lmsys/vicuna-7b-v1.5")
        
        # Load fine-tuned weights if available
        llm_weights_path = os.path.join(model_dir, "llm_model.pth")
        if os.path.exists(llm_weights_path):
            state_dict = torch.load(llm_weights_path, map_location=self.device)
            self.language_model.load_state_dict(state_dict)
        
        self.language_model.to(self.device)
        
    def load_projection_layer(self, model_dir):
        """Load projection layer for connecting vision and language"""
        import torch.nn as nn
        
        # Define projection layer
        self.projection = nn.Linear(
            self.vision_model.config.hidden_size,
            self.language_model.config.hidden_size
        )
        
        # Load weights
        projection_weights_path = os.path.join(model_dir, "projection_layer.pth")
        if os.path.exists(projection_weights_path):
            state_dict = torch.load(projection_weights_path, map_location=self.device)
            self.projection.load_state_dict(state_dict)
        
        self.projection.to(self.device)
        
    def preprocess_image(self, image_path):
        """Preprocess X-ray image for the model"""
        image = Image.open(image_path).convert('RGB')
        
        # Use CLIP processor for preprocessing
        inputs = self.vision_processor(images=image, return_tensors="pt")
        return inputs['pixel_values'].to(self.device)
        
    def generate_report(self, image_path, prompt="Generate a detailed radiology report for this chest X-ray:"):
        """Generate a report for an X-ray image"""
        self.vision_model.eval()
        self.language_model.eval()
        
        with torch.no_grad():
            # Process image
            image_tensor = self.preprocess_image(image_path)
            
            # Extract visual features
            vision_outputs = self.vision_model.vision_model(image_tensor)
            image_features = vision_outputs.last_hidden_state
            
            # Project to language model dimension
            projected_features = self.projection(image_features)
            
            # Prepare prompt
            text_inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            # Combine image features with text prompt
            # Note: This is a simplified version - actual implementation may vary
            inputs_embeds = self.language_model.get_input_embeddings()(text_inputs.input_ids)
            inputs_embeds = torch.cat([projected_features[:, 0:1, :], inputs_embeds], dim=1)
            
            # Generate report
            outputs = self.language_model.generate(
                inputs_embeds=inputs_embeds,
                max_length=512,
                num_beams=4,
                temperature=0.7,
                do_sample=True
            )
            
            report = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return report

# Training script
class XrayGPTTrainer:
    def __init__(self, base_model):
        self.model = base_model
        self.device = base_model.device
        
    def prepare_training_data(self, data_dir, batch_size=4):
        """Prepare data for fine-tuning"""
        from torch.utils.data import Dataset, DataLoader
        
        class XrayDataset(Dataset):
            def __init__(self, data_dir, vision_processor, tokenizer):
                self.data_dir = data_dir
                self.vision_processor = vision_processor
                self.tokenizer = tokenizer
                
                # Load metadata (adjust based on your dataset structure)
                self.metadata = pd.read_csv(os.path.join(data_dir, "metadata.csv"))
                
            def __len__(self):
                return len(self.metadata)
            
            def __getitem__(self, idx):
                row = self.metadata.iloc[idx]
                
                # Load image
                image_path = os.path.join(self.data_dir, row['path'])
                image = Image.open(image_path).convert('RGB')
                
                # Process image
                image_inputs = self.vision_processor(images=image, return_tensors="pt")
                
                # Process text
                report = row['report']
                text_inputs = self.tokenizer(
                    report,
                    truncation=True,
                    max_length=512,
                    padding='max_length',
                    return_tensors="pt"
                )
                
                return {
                    'pixel_values': image_inputs['pixel_values'].squeeze(0),
                    'input_ids': text_inputs['input_ids'].squeeze(0),
                    'attention_mask': text_inputs['attention_mask'].squeeze(0)
                }
        
        dataset = XrayDataset(data_dir, self.model.vision_processor, self.model.tokenizer)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return dataloader
    
    def train(self, train_dataloader, epochs=3, learning_rate=2e-5):
        """Fine-tune the model"""
        import torch.optim as optim
        
        # Freeze vision encoder and language model
        for param in self.model.vision_model.parameters():
            param.requires_grad = False
        for param in self.model.language_model.parameters():
            param.requires_grad = False
        
        # Only train projection layer
        optimizer = optim.AdamW(self.model.projection.parameters(), lr=learning_rate)
        
        self.model.projection.train()
        
        for epoch in range(epochs):
            total_loss = 0
            
            for batch in train_dataloader:
                # Move batch to device
                pixel_values = batch['pixel_values'].to(self.device)
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                # Forward pass
                with torch.no_grad():
                    vision_outputs = self.model.vision_model.vision_model(pixel_values)
                    image_features = vision_outputs.last_hidden_state
                
                # Project features
                projected_features = self.model.projection(image_features)
                
                # Prepare language model inputs
                inputs_embeds = self.model.language_model.get_input_embeddings()(input_ids)
                inputs_embeds[:, :projected_features.size(1), :] = projected_features
                
                # Language model forward pass
                outputs = self.model.language_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    labels=input_ids
                )
                
                loss = outputs.loss
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_dataloader)
            print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
    
    def save_model(self, save_dir):
        """Save the fine-tuned model"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save projection layer
        torch.save(
            self.model.projection.state_dict(),
            os.path.join(save_dir, "projection_layer.pth")
        )
        
        print(f"Model saved to {save_dir}")

# Dry run example
if __name__ == "__main__":
    # Dry run to test the model
    print("Starting dry run...")
    
    # Initialize the model from HuggingFace
    xraygpt = XrayGPTHuggingFace("khanhduong/xraygpt_vicuna")
    
    # Test inference with a sample image
    sample_image_path = "path/to/sample/xray.png"
    if os.path.exists(sample_image_path):
        report = xraygpt.generate_report(sample_image_path)
        print("Generated Report:")
        print(report)
    else:
        print("Please provide a valid X-ray image path")
    
    # Optional: Fine-tune the model on your dataset
    if False:  # Set to True to train
        trainer = XrayGPTTrainer(xraygpt)
        train_dataloader = trainer.prepare_training_data("data/raw/")
        trainer.train(train_dataloader, epochs=3)
        trainer.save_model("models/fine_tuned_xraygpt")