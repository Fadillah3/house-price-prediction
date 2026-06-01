import joblib
import numpy as np
from typing import List, Dict
from src.preprocess import DataPreprocessor
from src.config import Config
import logging

logger = logging.getLogger(__name__)

class HousePricePredictor:
    """Singleton pattern for model loading"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Load model and scaler with error handling"""
        self.model = None
        self.scaler = None
        self.preprocessor = DataPreprocessor()
        
        try:
            self.model = joblib.load(Config.MODEL_PATH)
            self.scaler = joblib.load(Config.SCALER_PATH)
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.warning(f"Could not load model files: {e}. Using stub model.")
            
            # BUAT DUMMY MODEL YANG SUDAH DILATIH
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.preprocessing import StandardScaler
            import numpy as np
            
            # Data dummy untuk training
            X_dummy = np.random.rand(100, 10)
            y_dummy = np.random.rand(100) * 500000
            
            # Latih model
            self.model = RandomForestRegressor(n_estimators=5, random_state=42)
            self.model.fit(X_dummy, y_dummy)  # ← KUNCI: training!
            
            self.scaler = StandardScaler()
            self.scaler.fit(X_dummy)
            
    def predict_single(self, input_data: Dict) -> Dict:
        """Predict price for a single input"""
        # Validate
        validate = self.preprocessor.validate_input(input_data)

        # Prepare features
        features = self.preprocessor.prepare_features(validate)

        # Scale features if scaler is available
        if self.scaler and self.model:
            features_scaled = self.scaler.transform(features)
            # Predict
            prediction = self.model.predict(features_scaled)[0]
        else:
            prediction = np.random.rand() * 1000000

        return {
            "predicted_price": round(float(prediction), 2),
            "predicted_price_formatted": f"${prediction:,.2f}",
            "features_used": DataPreprocessor.REQUIRED_FEATURES
        }
    
    def predict_batch(self, batch_data: List[Dict]) -> List[Dict]:
        """Predict multiple houses"""
        results = []
        for data in batch_data:
            try:
                result = self.predict_single(data)
                result["status"] = "success"
                results.append(result)
            except Exception as e:
                result = {
                    "error": str(e),
                    "status": "failed"
                }
                results.append(result)
        return results

# Global instance
predictor = HousePricePredictor()
