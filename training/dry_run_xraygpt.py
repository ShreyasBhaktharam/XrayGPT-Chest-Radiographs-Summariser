# dry_run_xraygpt.py
import sys
import os
from PIL import Image
import torch
from xraygpt_hf_integration import XrayGPTHuggingFace

def main():
    # Initialize the model
    print("Loading XrayGPT from HuggingFace...")
    model = XrayGPTHuggingFace("khanhduong/xraygpt_vicuna")
    
    # Test with a sample image
    # Use an example from your dataset
    sample_image_path = "/Users/shreyas/Downloads/XrayGPT-Chest-Radiographs-Summariser/data/raw/images/CXR1_1_IM-0001-3001.png"
    
    if not os.path.exists(sample_image_path):
        print(f"Sample image not found at {sample_image_path}")
        print("Please provide a valid X-ray image path")
        return
    
    print(f"Generating report for {sample_image_path}...")
    
    # Generate report
    report = model.generate_report(
        sample_image_path,
        prompt="Generate a detailed radiology report for this chest X-ray:"
    )
    
    print("\n=== Generated Report ===")
    print(report)
    print("========================\n")
    
    # Test with different prompts
    prompts = [
        "What abnormalities are visible in this chest X-ray?",
        "Describe the lung fields and cardiac silhouette in this image:",
        "Is there any evidence of pneumonia or consolidation?"
    ]
    
    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        response = model.generate_report(sample_image_path, prompt)
        print(f"Response: {response}")
        print("-" * 50)

if __name__ == "__main__":
    main()