import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    DATABASE_URL = os.environ.get('DATABASE_URL') or 'sqlite:///health_predictor.db'
    OLLAMA_BASE_URL = os.environ.get('OLLAMA_BASE_URL') or 'http://localhost:11434'
    OLLAMA_MODEL = os.environ.get('OLLAMA_MODEL') or 'llama2'
    
    # Health tracking configuration
    DAILY_TASKS = {
        'water_intake': {'target': 8, 'unit': 'glasses', 'points': 10},
        'fruit_consumption': {'target': 100, 'unit': 'grams', 'points': 15},
        'physical_activity': {'target': 2000, 'unit': 'steps', 'points': 20},
        'sleep_hours': {'target': 8, 'unit': 'hours', 'points': 15},
        'medication_taken': {'target': 1, 'unit': 'dose', 'points': 25},
        'stress_management': {'target': 1, 'unit': 'session', 'points': 10}
    }
    
    # Risk level thresholds
    RISK_THRESHOLDS = {
        'low': 0.25,
        'moderate': 0.50,
        'high': 0.75,
        'very_high': 1.0
    }
    
    # Model paths
    MODEL_DIR = 'models/saved_models/'
    DIABETES_RF_MODEL = os.path.join(MODEL_DIR, 'diabetes_rf.joblib')
    DIABETES_XGB_MODEL = os.path.join(MODEL_DIR, 'diabetes_xgb.joblib')
    
    # UI Configuration
    THEME_COLORS = {
        'primary': '#3B82F6',  # Medical blue
        'secondary': '#10B981',  # Wellness green
        'warning': '#F59E0B',   # Attention orange
        'danger': '#EF4444',    # High risk red
        'dark': '#1F2937',
        'light': '#F9FAFB'
    }