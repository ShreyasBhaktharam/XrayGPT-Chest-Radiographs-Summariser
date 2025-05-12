from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import base64
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import io
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response as StarletteResponse
import time
import sys
import os
import json
import argparse
import logging
from fastapi.concurrency import run_in_threadpool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add XrayGPT to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'XrayGPT'))

# Define the Pydantic models first - before any app initialization
class GenerateRequest(BaseModel):
    image: str  # Base64 encoded image
    message: str
    parameters: dict = Field(default_factory=dict)

class GenerateResponse(BaseModel):
    response: str

class FeedbackRequest(BaseModel):
    feedback_type: str  # "accept", "modify", or "reject"
    original_response: str
    modified_response: str = None

# Import XrayGPT modules
from xraygpt.common.config import Config
from xraygpt.common.registry import registry
from xraygpt.conversation.conversation import Chat, CONV_VISION

# Import models for registration
from xraygpt.models import *

app = FastAPI(
    title="XrayGPT API",
    description="API for interacting with XrayGPT model",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Prometheus metrics
REQUEST_COUNT = Counter("generate_requests_total", "Total number of generation requests")
REQUEST_LATENCY = Histogram("generate_request_latency_seconds", "Latency of generation requests")
GENERATION_ERRORS = Counter("generate_errors_total", "Total number of generation errors")
FEEDBACK_COUNTER = Counter(
    "feedback_total",
    "Total number of feedbacks received",
    ["feedback_type"]  # Label for different types of feedback
)

# ========================================
#            Helper Functions
# ========================================

def decode_image(base64_image):
    """Decode base64 image to PIL Image"""
    try:
        image_data = base64.b64decode(base64_image)
        return Image.open(io.BytesIO(image_data)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")

def save_feedback(feedback_data):
    """Save feedback to a file for future training"""
    os.makedirs("api_feedback_data", exist_ok=True)
    filename = f"api_feedback_data/feedback_{int(time.time())}.json"
    with open(filename, 'w') as f:
        json.dump(feedback_data, f, indent=2)

# ========================================
#              API Endpoints
# ========================================

@app.get("/metrics")
def metrics():
    return StarletteResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    REQUEST_COUNT.inc()
    start_time = time.time()
    logger.info("Request received")

    try:
        # Decode the base64 image (can also be run in threadpool if it becomes slow for huge images)
        logger.info("Decoding image")
        pil_image = await run_in_threadpool(decode_image, request.image)

        # Initialize conversation (this is likely fast, probably okay as is)
        logger.info("Initializing conversation")
        conversation = CONV_VISION.copy()
        img_list = [] # Keep img_list in the main thread context for now

        # Define a synchronous function to handle the blocking chat operations
        def process_and_generate_sync(pil_image, conversation, img_list, message, params):
            try:
                # Set CUDA device and clear cache before processing
                torch.cuda.empty_cache()
                torch.cuda.synchronize()  # Ensure all CUDA operations are complete
                
                logger.info("Processing image with model (in thread)")
                chat.upload_img(pil_image, conversation, img_list) # Modifies conversation and img_list in place

                logger.info("Adding user message (in thread)")
                chat.ask(message, conversation) # Modifies conversation in place

                logger.info("Extracting parameters (in thread)")
                num_beams = params.get("num_beams", 1)
                temperature = params.get("temperature", 0.7)  # Reduced default temperature
                max_new_tokens = params.get("max_new_tokens", 100)  # Reduced default tokens
                max_length = params.get("max_length", 500)  # Reduced default length

                logger.info("Generating response (in thread)")
                # NOTE: chat.answer might return a list, ensure you handle the output correctly
                llm_response = chat.answer(
                    conv=conversation,
                    img_list=img_list,
                    num_beams=num_beams,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                    max_length=max_length
                )
                
                # Ensure response is correctly extracted if chat.answer returns a list/tuple
                response = llm_response[0] if isinstance(llm_response, (list, tuple)) else llm_response
                
                # Clear CUDA cache after processing
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                return response
            except Exception as e:
                logger.error(f"Error in process_and_generate_sync: {str(e)}", exc_info=True)
                raise

        # Run the blocking operations in the thread pool with a semaphore to limit concurrent GPU operations
        response_text = await run_in_threadpool(
            process_and_generate_sync,
            pil_image=pil_image,
            conversation=conversation,
            img_list=img_list,
            message=request.message,
            params=request.parameters
        )

        logger.info("Response generated successfully")
        return GenerateResponse(response=response_text)

    except HTTPException as http_exc:
        # Re-raise HTTPExceptions directly
        raise http_exc
    except Exception as e:
        GENERATION_ERRORS.inc()
        logger.error(f"Error during processing: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

    finally:
        duration = time.time() - start_time
        logger.info(f"Request completed in {duration:.2f} seconds")
        REQUEST_LATENCY.observe(duration)

@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/feedback")
async def feedback(request: FeedbackRequest):
    # Record the feedback
    FEEDBACK_COUNTER.labels(feedback_type=request.feedback_type).inc()
    
    # Prepare feedback data with timestamp
    feedback_data = {
        "feedback_type": request.feedback_type,
        "original_response": request.original_response,
        "timestamp": time.time()
    }
    
    # Add modified response if provided
    if request.modified_response:
        feedback_data["modified_response"] = request.modified_response
    
    # Save feedback for future training
    save_feedback(feedback_data)
    
    return {"status": "feedback received"}

# ========================================
#         Model Initialization
# ========================================

# Only initialize the model when the script is run directly
if __name__ == "__main__":
    import uvicorn
    
    # Parse configuration
    parser = argparse.ArgumentParser(description="XrayGPT API Server")
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    
    # Parse the command line arguments
    args = parser.parse_args()
    
    try:
        cfg = Config(args)
        
        # Set CUDA device
        print(f"Setting default CUDA device to: cuda:{args.gpu_id}")
        torch.cuda.set_device(args.gpu_id)
        
        # Initialize model
        model_config = cfg.model_cfg
        model_config.device_8bit = args.gpu_id
        model_cls = registry.get_model_class(model_config.arch)
        model = model_cls.from_config(model_config).to(f'cuda:{args.gpu_id}')
        
        # Initialize visual processor
        vis_processor_cfg = cfg.datasets_cfg.openi.vis_processor.train
        vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
        
        # Initialize chat interface
        chat = Chat(model, vis_processor, device=f'cuda:{args.gpu_id}')
        print('Model Initialization Finished')
        
        # Start the server
        host = "0.0.0.0"
        port = 8800
        print(f"Starting XrayGPT API server on {host}:{port}")
        uvicorn.run(app, host=host, port=port)
        
    except Exception as e:
        print(f"Error during initialization: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
