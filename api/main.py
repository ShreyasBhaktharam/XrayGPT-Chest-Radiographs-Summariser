from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from typing import List, Optional
import numpy as np
from PIL import Image
import io
import onnxruntime
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI(title="XrayGPT API", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Prometheus metrics
Instrumentator().instrument(app).expose(app)

# Load ONNX model
ort_session = onnxruntime.InferenceSession("/models/xraygpt-latest.onnx")

class PredictionRequest(BaseModel):
    patient_id: str
    view_position: str
    patient_gender: str
    patient_age: int

class PredictionResponse(BaseModel):
    report: str
    confidence: float
    findings: List[str]
    impression: str

@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    metadata: PredictionRequest = None
):
    """Generate report for uploaded X-ray image"""
    try:
        # Read and preprocess image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        image = image.resize((224, 224))
        image_array = np.array(image).astype('float32') / 255.0
        image_array = np.transpose(image_array, (2, 0, 1))
        image_array = np.expand_dims(image_array, axis=0)
        
        # Prepare metadata
        metadata_array = np.array([
            metadata.patient_age / 100.0,
            1 if metadata.patient_gender == 'M' else 0,
            0 if metadata.view_position == 'PA' else 1
        ]).reshape(1, -1)
        
        # Run inference
        ort_inputs = {
            'image': image_array,
            'metadata': metadata_array
        }
        ort_outputs = ort_session.run(None, ort_inputs)
        
        # Process output
        report_text = ort_outputs[0][0]
        confidence = float(ort_outputs[1][0])
        
        return PredictionResponse(
            report=report_text,
            confidence=confidence,
            findings=["Finding 1", "Finding 2"],  # Parse from report
            impression="Impression text"  # Parse from report
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)