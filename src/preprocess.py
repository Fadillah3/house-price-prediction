import numpy as np
import pandas as pd
from typing import List, Dict

class DataPreprocessor:
    """Handle input validation and preprocessing"""
    
    REQUIRED_FEATURES = [
        'sqft_living', 'bedrooms', 'bathrooms', 'floors',
        'waterfront', 'view', 'condition', 'sqft_above',
        'sqft_basement', 'yr_built'
    ]
    
    @staticmethod
    def validate_input(data: Dict) -> Dict:
        """Validate input data types and ranges"""
        # Check required fields
        for feature in DataPreprocessor.REQUIRED_FEATURES:
            if feature not in data:
                raise ValueError(f"Missing required feature: {feature}")
        
        # Validate ranges
        validations = {
            'sqft_living': (300, 10000),
            'bedrooms': (1, 10),
            'bathrooms': (0.5, 8),
            'floors': (1, 4),
            'waterfront': (0, 1),
            'view': (0, 4),
            'condition': (1, 5),
            'sqft_above': (200, 8000),
            'sqft_basement': (0, 4000),
            'yr_built': (1800, 2024)
        }
        
        for feature, (min_val, max_val) in validations.items():
            value = data[feature]
            if not (min_val <= value <= max_val):
                raise ValueError(
                    f"{feature} must be between {min_val} and {max_val}, got {value}"
                )
        
        return data
    
    @staticmethod
    def prepare_features(data: Dict) -> np.ndarray:
        """Convert dict to numpy array in correct order"""
        features = [data[f] for f in DataPreprocessor.REQUIRED_FEATURES]
        return np.array(features).reshape(1, -1)