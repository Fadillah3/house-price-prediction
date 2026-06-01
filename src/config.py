import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    MODEL_PATH = os.getenv('MODEL_PATH', 'models/house_price_model.pkl')
    SCALER_PATH = os.getenv('SCALER_PATH', 'models/scaler.pkl')
    MAX_REQUEST_SIZE = int(os.getenv('MAX_REQUEST_SIZE', 100))
    CACHE_TTL = int(os.getenv('CACHE_TTL', 300))  # 5 minutes