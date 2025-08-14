#!/usr/bin/env python3
"""
Enhanced Diabetes Predictor - Machine Learning Models
Core ML model definitions and utilities
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
import xgboost as xgb
import joblib
import os
import logging
from typing import Dict, List, Tuple, Any, Optional
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class DiabetesMLModels:
    """Main class for diabetes prediction machine learning models"""
    
    def __init__(self):
        self.models = {}
        self.preprocessor = None
        self.feature_names = []
        self.is_trained = False
        
    def create_preprocessor(self, X: pd.DataFrame) -> Pipeline:
        """Create preprocessing pipeline"""
        try:
            # Identify numeric and categorical columns
            numeric_features = ['age', 'bmi', 'hba1c_level', 'blood_glucose_level']
            categorical_features = ['gender', 'smoking_history']
            binary_features = ['hypertension', 'heart_disease']
            
            # Ensure all expected features are present
            for feature in numeric_features + categorical_features + binary_features:
                if feature not in X.columns:
                    logger.warning(f"Feature {feature} not found in dataset")
            
            # Filter to existing features
            existing_numeric = [f for f in numeric_features if f in X.columns]
            existing_categorical = [f for f in categorical_features if f in X.columns]
            existing_binary = [f for f in binary_features if f in X.columns]
            
            # Create preprocessing steps
            from sklearn.compose import ColumnTransformer
            
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), existing_numeric),
                    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), existing_categorical),
                    ('bin', 'passthrough', existing_binary)
                ],
                remainder='drop'
            )
            
            self.feature_names = existing_numeric + existing_categorical + existing_binary
            return preprocessor
            
        except Exception as e:
            logger.error(f"Error creating preprocessor: {str(e)}")
            raise
    
    def create_random_forest_model(self) -> RandomForestClassifier:
        """Create Random Forest model with optimized hyperparameters"""
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
    
    def create_xgboost_model(self) -> xgb.XGBClassifier:
        """Create XGBoost model with optimized hyperparameters"""
        return xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss',
            use_label_encoder=False
        )
    
    def create_logistic_regression_model(self) -> LogisticRegression:
        """Create Logistic Regression model with optimized hyperparameters"""
        return LogisticRegression(
            random_state=42,
            max_iter=1000,
            class_weight='balanced',
            solver='liblinear'
        )
    
    def train_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Dict]:
        """Train all models and return performance metrics"""
        try:
            results = {}
            
            # Create preprocessor
            self.preprocessor = self.create_preprocessor(X)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Preprocess data
            X_train_processed = self.preprocessor.fit_transform(X_train)
            X_test_processed = self.preprocessor.transform(X_test)
            
            # Define models
            models_config = {
                'random_forest': self.create_random_forest_model(),
                'xgboost': self.create_xgboost_model(),
                'logistic_regression': self.create_logistic_regression_model()
            }
            
            # Train each model
            for name, model in models_config.items():
                logger.info(f"Training {name} model...")
                
                # Train model
                model.fit(X_train_processed, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test_processed)
                y_pred_proba = model.predict_proba(X_test_processed)[:, 1]
                
                # Calculate metrics
                metrics = self.calculate_metrics(y_test, y_pred, y_pred_proba)
                
                # Cross-validation score
                cv_scores = cross_val_score(model, X_train_processed, y_train, 
                                          cv=5, scoring='roc_auc')
                metrics['cv_auc_mean'] = cv_scores.mean()
                metrics['cv_auc_std'] = cv_scores.std()
                
                # Store model and results
                self.models[name] = model
                results[name] = metrics
                
                logger.info(f"{name} - AUC: {metrics['auc']:.3f}, Accuracy: {metrics['accuracy']:.3f}")
            
            self.is_trained = True
            return results
            
        except Exception as e:
            logger.error(f"Error training models: {str(e)}")
            raise
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         y_pred_proba: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive model performance metrics"""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred),
            'auc': roc_auc_score(y_true, y_pred_proba)
        }
        
        return metrics
    
    def predict_single(self, model_name: str, X: pd.DataFrame) -> Dict[str, Any]:
        """Make prediction with a single model"""
        try:
            if not self.is_trained:
                raise ValueError("Models must be trained before prediction")
            
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not found")
            
            model = self.models[model_name]
            
            # Preprocess input
            X_processed = self.preprocessor.transform(X)
            
            # Make prediction
            prediction = model.predict(X_processed)[0]
            probability = model.predict_proba(X_processed)[0, 1]
            
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
            logger.error(f"Error in prediction: {str(e)}")
            return {
                'model': model_name,
                'prediction': 0,
                'probability': 0.0,
                'risk_level': 'unknown',
                'confidence_interval': [0.0, 0.1],
                'error': str(e)
            }
    
    def predict_all_models(self, X: pd.DataFrame) -> Dict[str, Dict]:
        """Make predictions with all trained models"""
        predictions = {}
        
        for model_name in self.models.keys():
            predictions[model_name] = self.predict_single(model_name, X)
        
        # Add ensemble prediction
        if len(predictions) > 1:
            ensemble_pred = self.create_ensemble_prediction(predictions)
            predictions['ensemble'] = ensemble_pred
        
        return predictions
    
    def create_ensemble_prediction(self, predictions: Dict[str, Dict]) -> Dict[str, Any]:
        """Create ensemble prediction from multiple models"""
        try:
            # Get probabilities from all models (excluding any error cases)
            probs = []
            for pred in predictions.values():
                if 'error' not in pred:
                    probs.append(pred['probability'])
            
            if not probs:
                return {
                    'model': 'ensemble',
                    'prediction': 0,
                    'probability': 0.0,
                    'risk_level': 'unknown',
                    'confidence_interval': [0.0, 0.1]
                }
            
            # Calculate ensemble probability (average)
            ensemble_prob = np.mean(probs)
            ensemble_prediction = int(ensemble_prob > 0.5)
            
            return {
                'model': 'ensemble',
                'prediction': ensemble_prediction,
                'probability': float(ensemble_prob),
                'risk_level': self.get_risk_level(ensemble_prob),
                'confidence_interval': self.calculate_confidence_interval(ensemble_prob),
                'component_models': len(probs)
            }
            
        except Exception as e:
            logger.error(f"Error creating ensemble prediction: {str(e)}")
            return {
                'model': 'ensemble',
                'prediction': 0,
                'probability': 0.0,
                'risk_level': 'unknown',
                'confidence_interval': [0.0, 0.1],
                'error': str(e)
            }
    
    def get_risk_level(self, probability: float) -> str:
        """Convert probability to risk level category"""
        if probability < 0.25:
            return 'low'
        elif probability < 0.50:
            return 'moderate'
        elif probability < 0.75:
            return 'high'
        else:
            return 'very_high'
    
    def calculate_confidence_interval(self, probability: float, 
                                    confidence: float = 0.95) -> List[float]:
        """Calculate confidence interval for prediction"""
        # Simplified confidence interval calculation
        # In practice, you might use bootstrap or other methods
        margin = 0.1 * (1 - confidence + 0.05)
        lower = max(0.0, probability - margin)
        upper = min(1.0, probability + margin)
        return [float(lower), float(upper)]
    
    def get_feature_importance(self, model_name: str) -> Dict[str, float]:
        """Get feature importance for interpretable models"""
        try:
            if model_name not in self.models:
                return {}
            
            model = self.models[model_name]
            
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                
                # Get feature names after preprocessing
                feature_names = self.get_feature_names_after_preprocessing()
                
                importance_dict = {}
                for i, importance in enumerate(importances):
                    if i < len(feature_names):
                        importance_dict[feature_names[i]] = float(importance)
                
                # Sort by importance
                return dict(sorted(importance_dict.items(), 
                                 key=lambda x: x[1], reverse=True))
            
            return {}
            
        except Exception as e:
            logger.error(f"Error getting feature importance: {str(e)}")
            return {}
    
    def get_feature_names_after_preprocessing(self) -> List[str]:
        """Get feature names after preprocessing"""
        try:
            if self.preprocessor is None:
                return []
            
            # Get feature names from preprocessor
            feature_names = []
            
            # Numeric features (scaled)
            numeric_features = ['age', 'bmi', 'hba1c_level', 'blood_glucose_level']
            feature_names.extend(numeric_features)
            
            # Categorical features (one-hot encoded, drop first)
            categorical_features = ['gender', 'smoking_history']
            for cat_feature in categorical_features:
                if cat_feature == 'gender':
                    feature_names.append('gender_Male')  # Female is dropped
                elif cat_feature == 'smoking_history':
                    feature_names.extend(['smoking_former', 'smoking_never'])  # current is dropped
            
            # Binary features (passthrough)
            binary_features = ['hypertension', 'heart_disease']
            feature_names.extend(binary_features)
            
            return feature_names
            
        except Exception as e:
            logger.error(f"Error getting feature names: {str(e)}")
            return []
    
    def save_models(self, model_dir: str = 'models/saved_models/'):
        """Save trained models and preprocessor"""
        try:
            os.makedirs(model_dir, exist_ok=True)
            
            # Save models
            for name, model in self.models.items():
                model_path = os.path.join(model_dir, f'diabetes_{name}.joblib')
                joblib.dump(model, model_path)
                logger.info(f"Saved {name} model to {model_path}")
            
            # Save preprocessor
            if self.preprocessor:
                preprocessor_path = os.path.join(model_dir, 'preprocessor.joblib')
                joblib.dump(self.preprocessor, preprocessor_path)
                logger.info(f"Saved preprocessor to {preprocessor_path}")
            
            # Save metadata
            metadata = {
                'feature_names': self.feature_names,
                'is_trained': self.is_trained,
                'model_names': list(self.models.keys())
            }
            metadata_path = os.path.join(model_dir, 'metadata.joblib')
            joblib.dump(metadata, metadata_path)
            
            logger.info("All models saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving models: {str(e)}")
            raise
    
    def load_models(self, model_dir: str = 'models/saved_models/'):
        """Load trained models and preprocessor"""
        try:
            # Load metadata
            metadata_path = os.path.join(model_dir, 'metadata.joblib')
            if os.path.exists(metadata_path):
                metadata = joblib.load(metadata_path)
                self.feature_names = metadata.get('feature_names', [])
                self.is_trained = metadata.get('is_trained', False)
                model_names = metadata.get('model_names', [])
            else:
                model_names = ['random_forest', 'xgboost', 'logistic_regression']
            
            # Load models
            for name in model_names:
                model_path = os.path.join(model_dir, f'diabetes_{name}.joblib')
                if os.path.exists(model_path):
                    self.models[name] = joblib.load(model_path)
                    logger.info(f"Loaded {name} model")
            
            # Load preprocessor
            preprocessor_path = os.path.join(model_dir, 'preprocessor.joblib')
            if os.path.exists(preprocessor_path):
                self.preprocessor = joblib.load(preprocessor_path)
                logger.info("Loaded preprocessor")
            
            if self.models:
                logger.info(f"Successfully loaded {len(self.models)} models")
                return True
            else:
                logger.warning("No models found to load")
                return False
                
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            return False
    
    def hyperparameter_tuning(self, X: pd.DataFrame, y: pd.Series, 
                             model_name: str = 'random_forest') -> Dict[str, Any]:
        """Perform hyperparameter tuning for specified model"""
        try:
            logger.info(f"Starting hyperparameter tuning for {model_name}")
            
            # Preprocess data
            if self.preprocessor is None:
                self.preprocessor = self.create_preprocessor(X)
                X_processed = self.preprocessor.fit_transform(X)
            else:
                X_processed = self.preprocessor.transform(X)
            
            # Define parameter grids
            param_grids = {
                'random_forest': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 15, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },
                'xgboost': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 0.9, 1.0]
                },
                'logistic_regression': {
                    'C': [0.1, 1.0, 10.0],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'saga']
                }
            }
            
            if model_name not in param_grids:
                raise ValueError(f"Hyperparameter tuning not available for {model_name}")
            
            # Create base model
            if model_name == 'random_forest':
                base_model = RandomForestClassifier(random_state=42)
            elif model_name == 'xgboost':
                base_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
            elif model_name == 'logistic_regression':
                base_model = LogisticRegression(random_state=42, max_iter=1000)
            
            # Perform grid search
            grid_search = GridSearchCV(
                base_model,
                param_grids[model_name],
                cv=5,
                scoring='roc_auc',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_processed, y)
            
            # Store best model
            self.models[f'{model_name}_tuned'] = grid_search.best_estimator_
            
            results = {
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_,
                'best_model': grid_search.best_estimator_
            }
            
            logger.info(f"Best {model_name} parameters: {grid_search.best_params_}")
            logger.info(f"Best {model_name} score: {grid_search.best_score_:.3f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in hyperparameter tuning: {str(e)}")
            raise
    
    def model_comparison(self) -> pd.DataFrame:
        """Create comparison table of all models"""
        try:
            if not self.models:
                return pd.DataFrame()
            
            comparison_data = []
            
            for model_name, model in self.models.items():
                if hasattr(model, 'score'):  # Check if model has score method
                    # This would need test data - simplified for now
                    comparison_data.append({
                        'Model': model_name,
                        'Type': type(model).__name__,
                        'Parameters': len(str(model.get_params())),
                        'Trained': True
                    })
            
            return pd.DataFrame(comparison_data)
            
        except Exception as e:
            logger.error(f"Error creating model comparison: {str(e)}")
            return pd.DataFrame()

# Utility functions for backward compatibility
def create_diabetes_models():
    """Factory function to create DiabetesMLModels instance"""
    return DiabetesMLModels()

def train_diabetes_models(X: pd.DataFrame, y: pd.Series) -> DiabetesMLModels:
    """Train diabetes models and return trained instance"""
    models = DiabetesMLModels()
    models.train_models(X, y)
    return models