import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import xgboost as xgb
import joblib
import os
from typing import Dict, List, Tuple, Any
import logging
from config import Config

logger = logging.getLogger(__name__)

class PredictionService:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_names = []
        self.load_models()
    
    def load_models(self):
        """Load pre-trained models"""
        try:
            # Try to load existing models
            if os.path.exists(Config.DIABETES_RF_MODEL):
                self.models['random_forest'] = joblib.load(Config.DIABETES_RF_MODEL)
                logger.info("Loaded Random Forest model")
            
            if os.path.exists(Config.DIABETES_XGB_MODEL):
                self.models['xgboost'] = joblib.load(Config.DIABETES_XGB_MODEL)
                logger.info("Loaded XGBoost model")
            
            # Load scalers and encoders if they exist
            scaler_path = os.path.join(Config.MODEL_DIR, 'scaler.joblib')
            encoder_path = os.path.join(Config.MODEL_DIR, 'encoder.joblib')
            
            if os.path.exists(scaler_path):
                self.scalers['standard'] = joblib.load(scaler_path)
            
            if os.path.exists(encoder_path):
                self.encoders['categorical'] = joblib.load(encoder_path)
            
            # If no models exist, create default ones
            if not self.models:
                logger.warning("No pre-trained models found, creating default models")
                self.create_default_models()
                
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            self.create_default_models()
    
    def create_default_models(self):
        """Create default models with synthetic data"""
        try:
            # Create synthetic training data
            np.random.seed(42)
            n_samples = 1000
            
            # Generate synthetic features
            data = {
                'age': np.random.normal(45, 15, n_samples),
                'gender': np.random.choice(['Male', 'Female'], n_samples),
                'bmi': np.random.normal(26, 5, n_samples),
                'hypertension': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
                'heart_disease': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
                'smoking_history': np.random.choice(['never', 'former', 'current'], n_samples),
                'hba1c_level': np.random.normal(5.8, 1.2, n_samples),
                'blood_glucose_level': np.random.normal(120, 40, n_samples),
            }
            
            # Create target variable based on risk factors
            risk_scores = (
                (data['age'] - 30) / 50 +
                (data['bmi'] - 25) / 10 +
                data['hypertension'] * 0.3 +
                data['heart_disease'] * 0.4 +
                (data['hba1c_level'] - 5.7) / 2 +
                (data['blood_glucose_level'] - 100) / 100
            )
            
            # Add noise and create binary target
            risk_scores += np.random.normal(0, 0.2, n_samples)
            target = (risk_scores > 0.5).astype(int)
            
            df = pd.DataFrame(data)
            df['diabetes'] = target
            
            # Train models
            self.train_models(df)
            
        except Exception as e:
            logger.error(f"Error creating default models: {str(e)}")
    
    def preprocess_data(self, data: Dict, fit: bool = False) -> np.ndarray:
        """Preprocess input data for prediction"""
        try:
            # Convert to DataFrame
            if isinstance(data, dict):
                df = pd.DataFrame([data])
            else:
                df = pd.DataFrame(data)
            
            # Define expected features
            expected_features = [
                'age', 'gender', 'bmi', 'hypertension', 'heart_disease',
                'smoking_history', 'hba1c_level', 'blood_glucose_level'
            ]
            
            # Ensure all features are present
            for feature in expected_features:
                if feature not in df.columns:
                    if feature in ['hypertension', 'heart_disease']:
                        df[feature] = 0
                    elif feature == 'smoking_history':
                        df[feature] = 'never'
                    elif feature == 'gender':
                        df[feature] = 'Male'
                    else:
                        df[feature] = df.select_dtypes(include=[np.number]).mean().iloc[0]
            
            # Separate numerical and categorical features
            numerical_features = ['age', 'bmi', 'hba1c_level', 'blood_glucose_level']
            categorical_features = ['gender', 'smoking_history']
            binary_features = ['hypertension', 'heart_disease']
            
            # Process numerical features
            if fit:
                self.scalers['standard'] = StandardScaler()
                numerical_scaled = self.scalers['standard'].fit_transform(df[numerical_features])
            else:
                if 'standard' in self.scalers:
                    numerical_scaled = self.scalers['standard'].transform(df[numerical_features])
                else:
                    numerical_scaled = df[numerical_features].values
            
            # Process categorical features
            if fit:
                self.encoders['categorical'] = OneHotEncoder(drop='first', sparse_output=False)
                categorical_encoded = self.encoders['categorical'].fit_transform(df[categorical_features])
            else:
                if 'categorical' in self.encoders:
                    categorical_encoded = self.encoders['categorical'].transform(df[categorical_features])
                else:
                    # Simple encoding for default case
                    categorical_encoded = np.array([
                        [1 if df['gender'].iloc[0] == 'Male' else 0],
                        [1 if df['smoking_history'].iloc[0] == 'current' else 0]
                    ]).T
            
            # Combine all features
            features = np.hstack([
                numerical_scaled,
                categorical_encoded,
                df[binary_features].values
            ])
            
            return features
            
        except Exception as e:
            logger.error(f"Error preprocessing data: {str(e)}")
            # Return dummy features if preprocessing fails
            return np.zeros((1, 8))
    
    def train_models(self, df: pd.DataFrame):
        """Train all ML models"""
        try:
            # Prepare features and target
            X = self.preprocess_data(df.drop('diabetes', axis=1), fit=True)
            y = df['diabetes'].values
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Train Random Forest
            rf_model = RandomForestClassifier(
                n_estimators=100, random_state=42, max_depth=10
            )
            rf_model.fit(X_train, y_train)
            self.models['random_forest'] = rf_model
            
            # Train XGBoost
            xgb_model = xgb.XGBClassifier(
                n_estimators=100, random_state=42, max_depth=6
            )
            xgb_model.fit(X_train, y_train)
            self.models['xgboost'] = xgb_model
            
            # Train Logistic Regression
            lr_model = LogisticRegression(random_state=42, max_iter=1000)
            lr_model.fit(X_train, y_train)
            self.models['logistic_regression'] = lr_model
            
            # Evaluate models
            for name, model in self.models.items():
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)[:, 1]
                auc = roc_auc_score(y_test, y_prob)
                logger.info(f"{name} AUC: {auc:.3f}")
            
            # Save models
            self.save_models()
            
        except Exception as e:
            logger.error(f"Error training models: {str(e)}")
    
    def save_models(self):
        """Save trained models"""
        try:
            os.makedirs(Config.MODEL_DIR, exist_ok=True)
            
            # Save models
            for name, model in self.models.items():
                model_path = os.path.join(Config.MODEL_DIR, f'diabetes_{name}.joblib')
                joblib.dump(model, model_path)
            
            # Save preprocessors
            if 'standard' in self.scalers:
                joblib.dump(self.scalers['standard'], 
                           os.path.join(Config.MODEL_DIR, 'scaler.joblib'))
            
            if 'categorical' in self.encoders:
                joblib.dump(self.encoders['categorical'], 
                           os.path.join(Config.MODEL_DIR, 'encoder.joblib'))
            
            logger.info("Models saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving models: {str(e)}")
    
    def predict_single_model(self, data: Dict, model_name: str = 'random_forest') -> Dict:
        """Make prediction with single model"""
        try:
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not available")
            
            model = self.models[model_name]
            X = self.preprocess_data(data)
            
            # Get prediction and probability
            prediction = model.predict(X)[0]
            probability = model.predict_proba(X)[0, 1]
            
            # Calculate confidence interval (simplified)
            confidence_interval = self.calculate_confidence_interval(probability)
            
            # Determine risk level
            risk_level = self.get_risk_level(probability)
            
            return {
                'model': model_name,
                'prediction': int(prediction),
                'probability': float(probability),
                'risk_level': risk_level,
                'confidence_interval': confidence_interval
            }
            
        except Exception as e:
            logger.error(f"Error in single model prediction: {str(e)}")
            return {
                'model': model_name,
                'prediction': 0,
                'probability': 0.0,
                'risk_level': 'low',
                'confidence_interval': [0.0, 0.1],
                'error': str(e)
            }
    
    def predict_multiple_models(self, data: Dict) -> Dict:
        """Make predictions with all available models"""
        predictions = {}
        
        for model_name in self.models.keys():
            predictions[model_name] = self.predict_single_model(data, model_name)
        
        # Add ensemble prediction
        if len(predictions) > 1:
            probs = [pred['probability'] for pred in predictions.values()]
            ensemble_prob = np.mean(probs)
            ensemble_prediction = int(ensemble_prob > 0.5)
            
            predictions['ensemble'] = {
                'model': 'ensemble',
                'prediction': ensemble_prediction,
                'probability': float(ensemble_prob),
                'risk_level': self.get_risk_level(ensemble_prob),
                'confidence_interval': self.calculate_confidence_interval(ensemble_prob)
            }
        
        return predictions
    
    def get_risk_level(self, probability: float) -> str:
        """Convert probability to risk level"""
        if probability < Config.RISK_THRESHOLDS['low']:
            return 'low'
        elif probability < Config.RISK_THRESHOLDS['moderate']:
            return 'moderate'
        elif probability < Config.RISK_THRESHOLDS['high']:
            return 'high'
        else:
            return 'very_high'
    
    def calculate_confidence_interval(self, probability: float, confidence: float = 0.95) -> List[float]:
        """Calculate confidence interval for prediction"""
        # Simplified confidence interval calculation
        margin = 0.1 * (1 - confidence + 0.05)  # Adjust based on confidence
        lower = max(0.0, probability - margin)
        upper = min(1.0, probability + margin)
        return [float(lower), float(upper)]
    
    def get_feature_importance(self, model_name: str = 'random_forest') -> Dict:
        """Get feature importance for interpretability"""
        try:
            if model_name not in self.models:
                return {}
            
            model = self.models[model_name]
            
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                
                # Define feature names (should match preprocessing)
                feature_names = [
                    'age', 'bmi', 'hba1c_level', 'blood_glucose_level',
                    'gender_male', 'smoking_current', 'hypertension', 'heart_disease'
                ]
                
                importance_dict = {}
                for i, importance in enumerate(importances):
                    if i < len(feature_names):
                        importance_dict[feature_names[i]] = float(importance)
                
                return importance_dict
            
            return {}
            
        except Exception as e:
            logger.error(f"Error getting feature importance: {str(e)}")
            return {}
    
    def predict_cardiovascular_risk(self, data: Dict) -> Dict:
        """Predict cardiovascular risk (simplified model)"""
        try:
            # Calculate cardiovascular risk score based on known factors
            age = data.get('age', 40)
            bmi = data.get('bmi', 25)
            hypertension = data.get('hypertension', 0)
            heart_disease = data.get('heart_disease', 0)
            smoking = 1 if data.get('smoking_history') == 'current' else 0
            
            # Simple risk calculation
            risk_score = (
                (age - 40) / 40 * 0.3 +
                (bmi - 25) / 10 * 0.2 +
                hypertension * 0.25 +
                heart_disease * 0.4 +
                smoking * 0.2
            )
            
            probability = 1 / (1 + np.exp(-risk_score))  # Sigmoid
            
            return {
                'model': 'cardiovascular_risk',
                'probability': float(probability),
                'risk_level': self.get_risk_level(probability),
                'risk_factors': {
                    'age_factor': (age - 40) / 40 * 0.3,
                    'bmi_factor': (bmi - 25) / 10 * 0.2,
                    'hypertension_factor': hypertension * 0.25,
                    'heart_disease_factor': heart_disease * 0.4,
                    'smoking_factor': smoking * 0.2
                }
            }
            
        except Exception as e:
            logger.error(f"Error in cardiovascular risk prediction: {str(e)}")
            return {'model': 'cardiovascular_risk', 'probability': 0.0, 'risk_level': 'low'}
    
    def predict_metabolic_syndrome(self, data: Dict) -> Dict:
        """Predict metabolic syndrome risk"""
        try:
            bmi = data.get('bmi', 25)
            blood_glucose = data.get('blood_glucose_level', 100)
            hypertension = data.get('hypertension', 0)
            
            # Metabolic syndrome criteria (simplified)
            risk_factors = 0
            if bmi >= 30:  # Obesity
                risk_factors += 1
            if blood_glucose >= 100:  # Elevated glucose
                risk_factors += 1
            if hypertension:  # High blood pressure
                risk_factors += 1
            
            probability = risk_factors / 3.0  # Normalize to 0-1
            
            return {
                'model': 'metabolic_syndrome',
                'probability': float(probability),
                'risk_level': self.get_risk_level(probability),
                'risk_factors_count': risk_factors
            }
            
        except Exception as e:
            logger.error(f"Error in metabolic syndrome prediction: {str(e)}")
            return {'model': 'metabolic_syndrome', 'probability': 0.0, 'risk_level': 'low'}