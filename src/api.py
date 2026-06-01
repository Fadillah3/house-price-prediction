from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
import time
import logging

from src.predict import predictor
from src.config import Config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="House Price Prediction API",
    description="Predict house prices using ML model",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class HouseFeatures(BaseModel):
    sqft_living: int = Field(..., ge=300, le=10000, description="Living area in sqft")
    bedrooms: int = Field(..., ge=1, le=10, description="Number of bedrooms")
    bathrooms: float = Field(..., ge=0.5, le=8, description="Number of bathrooms")
    floors: float = Field(..., ge=1, le=4, description="Number of floors")
    waterfront: int = Field(..., ge=0, le=1, description="Waterfront view (0/1)")
    view: int = Field(..., ge=0, le=4, description="View rating")
    condition: int = Field(..., ge=1, le=5, description="House condition")
    sqft_above: int = Field(..., ge=200, le=8000, description="Above ground area")
    sqft_basement: int = Field(..., ge=0, le=4000, description="Basement area")
    yr_built: int = Field(..., ge=1800, le=2024, description="Year built")
    
    class Config:
        json_schema_extra = {
            "example": {
                "sqft_living": 2000,
                "bedrooms": 3,
                "bathrooms": 2.5,
                "floors": 2,
                "waterfront": 0,
                "view": 2,
                "condition": 3,
                "sqft_above": 1500,
                "sqft_basement": 500,
                "yr_built": 1995
            }
        }

class PredictionResponse(BaseModel):
    predicted_price: float
    predicted_price_formatted: str
    features_used: List[str]
    latency_ms: Optional[float] = None

class BatchPredictionRequest(BaseModel):
    houses: List[HouseFeatures]

class BatchPredictionResponse(BaseModel):
    predictions: List[dict]
    total_latency_ms: float

# Health check
@app.get("/")
async def root():
    return {
        "service": "House Price Prediction API",
        "status": "healthy",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict", response_model=PredictionResponse)
async def predict(house: HouseFeatures):
    """Predict single house price"""
    start_time = time.time()
    
    try:
        result = predictor.predict_single(house.dict())
        latency = (time.time() - start_time) * 1000
        
        logger.info(f"Prediction completed in {latency:.2f}ms")
        
        return PredictionResponse(
            **result,
            latency_ms=round(latency, 2)
        )
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """Predict multiple house prices"""
    start_time = time.time()
    
    results = predictor.predict_batch([house.dict() for house in request.houses])
    total_latency = (time.time() - start_time) * 1000
    
    return BatchPredictionResponse(
        predictions=results,
        total_latency_ms=round(total_latency, 2)
    )

@app.get("/model/info")
async def model_info():
    """Get model information"""
    return {
        "model_type": "RandomForestRegressor",
        "features": predictor.preprocessor.REQUIRED_FEATURES,
        "endpoints": ["/predict", "/predict/batch", "/health"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )