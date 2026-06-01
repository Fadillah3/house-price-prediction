import pytest
from src.preprocess import DataPreprocessor

def test_valid_input():
    preprocessor = DataPreprocessor()
    valid_data = {
        'sqft_living': 2000,
        'bedrooms': 3,
        'bathrooms': 2.5,
        'floors': 2,
        'waterfront': 0,
        'view': 2,
        'condition': 3,
        'sqft_above': 1500,
        'sqft_basement': 500,
        'yr_built': 1995
    }
    
    result = preprocessor.validate_input(valid_data)
    assert result == valid_data

def test_missing_feature():
    preprocessor = DataPreprocessor()
    invalid_data = {
        'sqft_living': 2000,
        'bedrooms': 3
    }
    
    with pytest.raises(ValueError, match="Missing required feature"):
        preprocessor.validate_input(invalid_data)

def test_out_of_range():
    preprocessor = DataPreprocessor()
    invalid_data = {
        'sqft_living': 2000,
        'bedrooms': 20,  # Too many bedrooms
        'bathrooms': 2.5,
        'floors': 2,
        'waterfront': 0,
        'view': 2,
        'condition': 3,
        'sqft_above': 1500,
        'sqft_basement': 500,
        'yr_built': 1995
    }
    
    with pytest.raises(ValueError):
        preprocessor.validate_input(invalid_data)