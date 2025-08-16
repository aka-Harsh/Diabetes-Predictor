#!/usr/bin/env python3
"""
Enhanced Diabetes Predictor - Model Training Script
"""

import pandas as pd
import numpy as np
import os
import sys
import logging
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from services.prediction_service import PredictionService

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def create_synthetic_dataset(n_samples=10000, save_path=None):
    """Create a synthetic diabetes dataset for training"""
    logger.info(f"Creating synthetic dataset with {n_samples} samples...")
    
    np.random.seed(42)  # For reproducibility
    
    # Generate correlated features that realistically predict diabetes
    data = {}
    
    # Age distribution (skewed towards older adults)
    data['age'] = np.random.gamma(2, 20) + 18
    data['age'] = np.clip(data['age'], 18, 95)
    
    # Gender (roughly equal distribution)
    data['gender'] = np.random.choice(['Male', 'Female'], n_samples, p=[0.48, 0.52])
    
    # BMI (correlated with age and gender)
    base_bmi = np.random.normal(26, 5, n_samples)
    age_effect = (data['age'] - 40) / 40 * 2  # BMI increases with age
    gender_effect = np.where(data['gender'] == 'Male', 1, 0)  # Males slightly higher BMI
    data['bmi'] = base_bmi + age_effect + gender_effect
    data['bmi'] = np.clip(data['bmi'], 15, 50)
    
    # Hypertension (increases with age and BMI)
    hypertension_prob = (
        0.1 +  # base rate
        (data['age'] - 30) / 60 * 0.4 +  # age effect
        np.maximum(data['bmi'] - 25, 0) / 15 * 0.3  # BMI effect
    )
    hypertension_prob = np.clip(hypertension_prob, 0, 0.8)
    data['hypertension'] = np.random.binomial(1, hypertension_prob)
    
    # Heart disease (correlated with age, BMI, and hypertension)
    heart_disease_prob = (
        0.05 +  # base rate
        (data['age'] - 40) / 50 * 0.2 +  # age effect
        data['hypertension'] * 0.15 +  # hypertension effect
        np.maximum(data['bmi'] - 30, 0) / 20 * 0.1  # obesity effect
    )
    heart_disease_prob = np.clip(heart_disease_prob, 0, 0.5)
    data['heart_disease'] = np.random.binomial(1, heart_disease_prob)
    
    # Smoking history
    smoking_probs = [0.6, 0.25, 0.15]  # never, former, current
    data['smoking_history'] = np.random.choice(['never', 'former', 'current'], 
                                               n_samples, p=smoking_probs)
    
    # HbA1c level (key diabetes indicator)
    base_hba1c = np.random.normal(5.4, 0.8, n_samples)
    age_effect = (data['age'] - 40) / 60 * 0.5
    bmi_effect = np.maximum(data['bmi'] - 25, 0) / 15 * 0.8
    data['hba1c_level'] = base_hba1c + age_effect + bmi_effect
    data['hba1c_level'] = np.clip(data['hba1c_level'], 3.5, 15.0)
    
    # Blood glucose level (correlated with HbA1c)
    glucose_base = (data['hba1c_level'] - 5.7) * 35 + 100  # Rough conversion
    glucose_noise = np.random.normal(0, 15, n_samples)  # Random variation
    data['blood_glucose_level'] = glucose_base + glucose_noise
    data['blood_glucose_level'] = np.clip(data['blood_glucose_level'], 70, 400)
    
    # Create diabetes target variable based on realistic risk factors
    diabetes_risk = (
        # Age factor
        np.maximum(data['age'] - 45, 0) / 30 * 0.2 +
        
        # BMI factor
        np.maximum(data['bmi'] - 25, 0) / 15 * 0.25 +
        
        # HbA1c factor (most important)
        np.maximum(data['hba1c_level'] - 5.7, 0) / 3 * 0.35 +
        
        # Blood glucose factor
        np.maximum(data['blood_glucose_level'] - 100, 0) / 100 * 0.15 +
        
        # Comorbidities
        data['hypertension'] * 0.08 +
        data['heart_disease'] * 0.12 +
        
        # Smoking
        np.where(data['smoking_history'] == 'current', 0.05, 0) +
        
        # Random noise
        np.random.normal(0, 0.1, n_samples)
    )
    
    # Convert risk to binary diabetes outcome
    diabetes_prob = 1 / (1 + np.exp(-3 * (diabetes_risk - 0.5)))  # Sigmoid
    data['diabetes'] = np.random.binomial(1, diabetes_prob)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Log statistics
    logger.info("Dataset Statistics:")
    logger.info(f"Total samples: {len(df)}")
    logger.info(f"Diabetes prevalence: {df['diabetes'].mean():.3f}")
    logger.info(f"Age range: {df['age'].min():.1f} - {df['age'].max():.1f}")
    logger.info(f"BMI range: {df['bmi'].min():.1f} - {df['bmi'].max():.1f}")
    logger.info(f"HbA1c range: {df['hba1c_level'].min():.1f} - {df['hba1c_level'].max():.1f}")
    
    # Save dataset
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
        logger.info(f"Dataset saved to {save_path}")
    
    return df

def train_models():
    """Train and save all machine learning models"""
    logger.info("Starting model training process...")
    
    # Check if dataset exists, create if not
    dataset_path = 'data/diabetes_data.csv'
    
    if not os.path.exists(dataset_path):
        logger.info("Dataset not found, creating synthetic dataset...")
        df = create_synthetic_dataset(10000, dataset_path)
    else:
        logger.info(f"Loading existing dataset from {dataset_path}")
        df = pd.read_csv(dataset_path)
    
    # Initialize prediction service
    prediction_service = PredictionService()
    
    # Train models
    try:
        prediction_service.train_models(df)
        logger.info("Model training completed successfully!")
        
        # Test predictions with sample data
        test_sample = {
            'age': 45,
            'gender': 'Male',
            'bmi': 28.5,
            'hypertension': 1,
            'heart_disease': 0,
            'smoking_history': 'former',
            'hba1c_level': 6.2,
            'blood_glucose_level': 140
        }
        
        logger.info("Testing trained models with sample data...")
        predictions = prediction_service.predict_multiple_models(test_sample)
        
        for model_name, prediction in predictions.items():
            logger.info(f"{model_name}: {prediction['risk_level']} risk "
                       f"({prediction['probability']:.3f} probability)")
        
        return True
        
    except Exception as e:
        logger.error(f"Model training failed: {str(e)}")
        return False

def evaluate_models():
    """Evaluate model performance"""
    logger.info("Evaluating model performance...")
    
    try:
        # Load test dataset
        df = pd.read_csv('data/diabetes_data.csv')
        
        # Split into train/test
        from sklearn.model_selection import train_test_split
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, 
                                           stratify=df['diabetes'])
        
        # Initialize service
        prediction_service = PredictionService()
        
        # Test predictions on test set
        test_features = test_df.drop('diabetes', axis=1)
        test_labels = test_df['diabetes']
        
        results = {}
        
        for model_name in prediction_service.models.keys():
            predictions = []
            probabilities = []
            
            for _, row in test_features.iterrows():
                pred = prediction_service.predict_single_model(row.to_dict(), model_name)
                predictions.append(pred['prediction'])
                probabilities.append(pred['probability'])
            
            # Calculate metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
            
            accuracy = accuracy_score(test_labels, predictions)
            precision = precision_score(test_labels, predictions)
            recall = recall_score(test_labels, predictions)
            f1 = f1_score(test_labels, predictions)
            auc = roc_auc_score(test_labels, probabilities)
            
            results[model_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'auc_score': auc
            }
            
            logger.info(f"\n{model_name.upper()} Performance:")
            logger.info(f"  Accuracy:  {accuracy:.4f}")
            logger.info(f"  Precision: {precision:.4f}")
            logger.info(f"  Recall:    {recall:.4f}")
            logger.info(f"  F1 Score:  {f1:.4f}")
            logger.info(f"  AUC Score: {auc:.4f}")
        
        return results
        
    except Exception as e:
        logger.error(f"Model evaluation failed: {str(e)}")
        return None

def main():
    """Main training script"""
    logger.info("="*50)
    logger.info("Enhanced Diabetes Predictor - Model Training")
    logger.info("="*50)
    
    try:
        # Create necessary directories
        os.makedirs('data', exist_ok=True)
        os.makedirs('models/saved_models', exist_ok=True)
        
        # Train models
        success = train_models()
        
        if success:
            logger.info("✅ Model training completed successfully!")
            
            # Evaluate models
            results = evaluate_models()
            if results:
                logger.info("✅ Model evaluation completed!")
            
        else:
            logger.error("❌ Model training failed!")
            return 1
            
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return 1
    
    logger.info("Training process completed!")
    return 0

if __name__ == "__main__":
    exit(main())