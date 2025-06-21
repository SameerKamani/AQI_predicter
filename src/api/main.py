from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import sys
import os
from datetime import datetime, timedelta

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from predict import PM10Predictor

# Initialize FastAPI app
app = FastAPI(
    title="PM10 Prediction API",
    description="AI-powered PM10 air quality prediction system",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the predictor
try:
    predictor = PM10Predictor()
    print("✅ PM10 Predictor loaded successfully in API")
except Exception as e:
    print(f"❌ Error loading predictor: {e}")
    predictor = None

# Pydantic models for request/response
class SinglePredictionRequest(BaseModel):
    pm10_history: List[float]
    temperature: float = 20.0
    timestamp: Optional[str] = None

class SinglePredictionResponse(BaseModel):
    prediction: float
    confidence: str
    timestamp: str
    model_info: dict

class BatchPredictionRequest(BaseModel):
    pm10_history: List[float]
    temperatures: List[float]
    timestamps: Optional[List[str]] = None

class BatchPredictionResponse(BaseModel):
    predictions: List[float]
    timestamps: List[str]
    confidence: str
    model_info: dict

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    timestamp: str

@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information"""
    return {
        "message": "PM10 Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "predict_batch": "/predict/batch",
            "docs": "/docs"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if predictor else "unhealthy",
        model_loaded=predictor is not None,
        timestamp=datetime.now().isoformat()
    )

@app.post("/predict", response_model=SinglePredictionResponse)
async def predict_single(request: SinglePredictionRequest):
    """Predict PM10 for a single point in time"""
    if not predictor:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Parse timestamp if provided
        timestamp = None
        if request.timestamp:
            timestamp = datetime.fromisoformat(request.timestamp.replace('Z', '+00:00'))
        
        # Make prediction
        prediction = predictor.predict_single(
            request.pm10_history,
            request.temperature,
            timestamp
        )
        
        # Get confidence
        confidence = predictor.get_prediction_confidence(request.pm10_history)
        
        return SinglePredictionResponse(
            prediction=float(prediction),
            confidence=confidence,
            timestamp=datetime.now().isoformat(),
            model_info={
                "model_type": "Linear Regression",
                "r2_score": 1.0000,
                "features_used": len(predictor.features)
            }
        )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """Predict PM10 for multiple points in time"""
    if not predictor:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Parse timestamps if provided
        timestamps = None
        if request.timestamps:
            timestamps = [
                datetime.fromisoformat(ts.replace('Z', '+00:00'))
                for ts in request.timestamps
            ]
        
        # Make predictions
        predictions = predictor.predict_multiple(
            request.pm10_history,
            request.temperatures,
            timestamps
        )
        
        # Get confidence
        confidence = predictor.get_prediction_confidence(request.pm10_history)
        
        # Generate timestamps if not provided
        if not timestamps:
            timestamps = [
                datetime.now() + timedelta(hours=i)
                for i in range(len(request.temperatures))
            ]
        
        return BatchPredictionResponse(
            predictions=[float(p) for p in predictions],
            timestamps=[ts.isoformat() for ts in timestamps],
            confidence=confidence,
            model_info={
                "model_type": "Linear Regression",
                "r2_score": 1.0000,
                "features_used": len(predictor.features)
            }
        )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/model/info")
async def model_info():
    """Get information about the loaded model"""
    if not predictor:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": "Linear Regression",
        "r2_score": 1.0000,
        "features": predictor.features,
        "features_count": len(predictor.features),
        "model_file": predictor.model_file,
        "scaler_file": predictor.scaler_file
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 