import pytest
from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] == True

def test_predict_valid():
    test_house = {
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
    
    response = client.post("/predict", json=test_house)
    assert response.status_code == 200
    
    data = response.json()
    assert "predicted_price" in data
    assert "predicted_price_formatted" in data
    assert data["predicted_price"] > 0

def test_predict_invalid():
    test_house = {
        "sqft_living": 2000,
        "bedrooms": 20,  # Invalid
        "bathrooms": 2.5,
        "floors": 2,
        "waterfront": 0,
        "view": 2,
        "condition": 3,
        "sqft_above": 1500,
        "sqft_basement": 500,
        "yr_built": 1995
    }
    
    response = client.post("/predict", json=test_house)
    assert response.status_code == 400

def test_batch_predict():
    test_houses = [
        {
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
        },
        {
            "sqft_living": 3000,
            "bedrooms": 4,
            "bathrooms": 3.5,
            "floors": 2,
            "waterfront": 1,
            "view": 4,
            "condition": 5,
            "sqft_above": 2500,
            "sqft_basement": 500,
            "yr_built": 2005
        }
    ]
    
    response = client.post("/predict/batch", json={"houses": test_houses})
    assert response.status_code == 422
    
    data = response.json()
    assert len(data["predictions"]) == 2