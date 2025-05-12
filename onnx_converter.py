import os
import torch
import onnx
import onnxruntime as ort
from datetime import datetime
from omegaconf import OmegaConf
from xraygpt.models.mini_gpt4 import MiniGPT4

class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # Create the expected input format with both image and caption
        samples = {
            "image": x,
            "caption": ["Describe this X-ray image."]  # Add a default caption
        }
        return self.model(samples)

def convert_to_onnx(model_path, onnx_path, config_path="xraygpt/configs/models/xraygpt.yaml"):
    """
    Convert a PyTorch model to ONNX format
    
    Args:
        model_path (str): Path to the PyTorch model file
        onnx_path (str): Path where the ONNX model will be saved
        config_path (str): Path to the model configuration file
    """
    print(f"\nConverting {model_path} to ONNX format...")
    
    # Load model configuration
    cfg = OmegaConf.load(config_path)
    model_cfg = cfg.model
    
    # Initialize model
    model = MiniGPT4(
        vit_model=model_cfg.get("vit_model", "eva_clip_g"),
        img_size=model_cfg.get("image_size", 224),
        drop_path_rate=model_cfg.get("drop_path_rate", 0),
        use_grad_checkpoint=model_cfg.get("use_grad_checkpoint", False),
        vit_precision=model_cfg.get("vit_precision", "fp32"),
        freeze_vit=model_cfg.get("freeze_vit", True),
        freeze_qformer=model_cfg.get("freeze_qformer", True),
        num_query_token=model_cfg.get("num_query_token", 32),
        llama_model=model_cfg.get("llama_model", ""),
        prompt_path=model_cfg.get("prompt", ""),
        prompt_template=model_cfg.get("prompt_template", ""),
        max_txt_len=model_cfg.get("max_txt_len", 32),
        low_resource=model_cfg.get("low_resource", False)
    )
    
    # Load state dict
    state_dict = torch.load(model_path, map_location="cpu")
    if isinstance(state_dict, dict):
        if "model" in state_dict:
            state_dict = state_dict["model"]
        elif "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
    
    # Remove module prefix if present
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    
    # Load state dict with strict=False to allow partial loading
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
    print("Missing keys:", len(missing_keys))
    print("Unexpected keys:", len(unexpected_keys))
    
    # Convert model to float32
    model = model.float()
    model.eval()
    
    # Wrap the model for ONNX export
    wrapped_model = ModelWrapper(model)
    wrapped_model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)
    
    # Export the model
    torch.onnx.export(
        wrapped_model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=20,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        },
        #use_external_data_format=True
    )
    
    # Verify the ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print(f"ONNX model saved to {onnx_path}")
    print("ONNX model verification successful!")

def main():
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Get current timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Convert xraygpt model
    xraygpt_model_path = "/home/cc/xraygpt_pretrained1.pth"
    xraygpt_onnx_path = f"models/xraygpt_{timestamp}.onnx"
    convert_to_onnx(xraygpt_model_path, xraygpt_onnx_path)
    
    # Convert minigpt4 model
    minigpt4_model_path = "/home/cc/pretrained_minigpt4_7b.pth"
    minigpt4_onnx_path = f"models/minigpt4_{timestamp}.onnx"
    convert_to_onnx(minigpt4_model_path, minigpt4_onnx_path)

if __name__ == "__main__":
    main() 
