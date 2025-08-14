from typing import Dict, List, Any, Tuple
import re
import logging

logger = logging.getLogger(__name__)

class InputValidator:
    def __init__(self):
        self.validation_rules = {
            'age': {'min': 18, 'max': 120, 'required': True},
            'gender': {'options': ['Male', 'Female'], 'required': True},
            'bmi': {'min': 10, 'max': 60, 'required': True},
            'hypertension': {'type': 'boolean', 'required': False},
            'heart_disease': {'type': 'boolean', 'required': False},
            'smoking_history': {'options': ['never', 'former', 'current'], 'required': False},
            'hba1c_level': {'min': 3.0, 'max': 20.0, 'required': False},
            'blood_glucose_level': {'min': 50, 'max': 500, 'required': False}
        }
    
    def validate_prediction_input(self, data: Dict) -> Dict:
        """Validate input data for prediction"""
        try:
            errors = []
            warnings = []
            cleaned_data = {}
            
            # Check required fields
            for field, rules in self.validation_rules.items():
                if rules.get('required', False) and field not in data:
                    errors.append(f"Missing required field: {field}")
                    continue
                
                if field in data:
                    value = data[field]
                    validation_result = self._validate_field(field, value, rules)
                    
                    if validation_result['valid']:
                        cleaned_data[field] = validation_result['value']
                        if validation_result.get('warning'):
                            warnings.append(validation_result['warning'])
                    else:
                        errors.extend(validation_result['errors'])
            
            # Additional validation logic
            additional_errors, additional_warnings = self._validate_field_combinations(cleaned_data)
            errors.extend(additional_errors)
            warnings.extend(additional_warnings)
            
            return {
                'valid': len(errors) == 0,
                'errors': errors,
                'warnings': warnings,
                'cleaned_data': cleaned_data
            }
            
        except Exception as e:
            logger.error(f"Validation error: {str(e)}")
            return {
                'valid': False,
                'errors': [f"Validation failed: {str(e)}"],
                'warnings': [],
                'cleaned_data': {}
            }
    
    def _validate_field(self, field: str, value: Any, rules: Dict) -> Dict:
        """Validate individual field"""
        try:
            # Handle None values
            if value is None:
                if rules.get('required', False):
                    return {'valid': False, 'errors': [f"{field} is required"]}
                else:
                    return {'valid': True, 'value': None}
            
            # Type conversion and validation
            if rules.get('type') == 'boolean':
                return self._validate_boolean(field, value)
            elif 'options' in rules:
                return self._validate_options(field, value, rules['options'])
            elif 'min' in rules or 'max' in rules:
                return self._validate_numeric(field, value, rules)
            else:
                return self._validate_string(field, value)
                
        except Exception as e:
            return {'valid': False, 'errors': [f"Error validating {field}: {str(e)}"]}
    
    def _validate_boolean(self, field: str, value: Any) -> Dict:
        """Validate boolean field"""
        if isinstance(value, bool):
            return {'valid': True, 'value': value}
        elif isinstance(value, str):
            if value.lower() in ['true', '1', 'yes', 'on']:
                return {'valid': True, 'value': True}
            elif value.lower() in ['false', '0', 'no', 'off']:
                return {'valid': True, 'value': False}
            else:
                return {'valid': False, 'errors': [f"{field} must be true or false"]}
        elif isinstance(value, (int, float)):
            return {'valid': True, 'value': bool(value)}
        else:
            return {'valid': False, 'errors': [f"{field} must be a boolean value"]}
    
    def _validate_options(self, field: str, value: Any, options: List[str]) -> Dict:
        """Validate field with predefined options"""
        if str(value) in options:
            return {'valid': True, 'value': str(value)}
        else:
            return {
                'valid': False, 
                'errors': [f"{field} must be one of: {', '.join(options)}"]
            }
    
    def _validate_numeric(self, field: str, value: Any, rules: Dict) -> Dict:
        """Validate numeric field"""
        try:
            # Convert to float
            if isinstance(value, str):
                numeric_value = float(value.replace(',', ''))
            else:
                numeric_value = float(value)
            
            # Check range
            min_val = rules.get('min')
            max_val = rules.get('max')
            warning = None
            
            if min_val is not None and numeric_value < min_val:
                return {
                    'valid': False, 
                    'errors': [f"{field} must be at least {min_val}"]
                }
            
            if max_val is not None and numeric_value > max_val:
                return {
                    'valid': False, 
                    'errors': [f"{field} must be no more than {max_val}"]
                }
            
            # Add warnings for unusual but valid values
            if field == 'age' and numeric_value > 90:
                warning = "Age over 90 detected - please verify"
            elif field == 'bmi' and numeric_value < 15:
                warning = "Very low BMI detected - please verify"
            elif field == 'bmi' and numeric_value > 45:
                warning = "Very high BMI detected - please verify"
            elif field == 'blood_glucose_level' and numeric_value > 300:
                warning = "Very high blood glucose - please seek immediate medical attention"
            elif field == 'hba1c_level' and numeric_value > 12:
                warning = "Very high HbA1c - please consult healthcare provider"
            
            return {
                'valid': True, 
                'value': numeric_value,
                'warning': warning
            }
            
        except (ValueError, TypeError):
            return {
                'valid': False, 
                'errors': [f"{field} must be a valid number"]
            }
    
    def _validate_string(self, field: str, value: Any) -> Dict:
        """Validate string field"""
        str_value = str(value).strip()
        
        if len(str_value) == 0:
            return {'valid': False, 'errors': [f"{field} cannot be empty"]}
        
        if len(str_value) > 255:
            return {'valid': False, 'errors': [f"{field} is too long (max 255 characters)"]}
        
        # Basic sanitization
        str_value = re.sub(r'[<>\"\'&]', '', str_value)
        
        return {'valid': True, 'value': str_value}
    
    def _validate_field_combinations(self, data: Dict) -> Tuple[List[str], List[str]]:
        """Validate combinations of fields for logical consistency"""
        errors = []
        warnings = []
        
        try:
            # Age and health conditions consistency
            age = data.get('age')
            if age and age < 25:
                if data.get('heart_disease'):
                    warnings.append("Heart disease is uncommon in people under 25")
                if data.get('hypertension'):
                    warnings.append("Hypertension is uncommon in people under 25")
            
            # BMI and health consistency
            bmi = data.get('bmi')
            if bmi:
                if bmi < 18.5:
                    warnings.append("BMI indicates underweight - diabetes risk may differ")
                elif bmi > 35 and not data.get('hypertension'):
                    warnings.append("High BMI often correlates with hypertension")
            
            # Blood glucose and HbA1c consistency
            glucose = data.get('blood_glucose_level')
            hba1c = data.get('hba1c_level')
            
            if glucose and hba1c:
                # Rough correlation check
                estimated_hba1c = (glucose + 46.7) / 28.7
                if abs(hba1c - estimated_hba1c) > 2:
                    warnings.append("Blood glucose and HbA1c values may not be consistent")
            
            # Critical values requiring immediate attention
            if glucose and glucose > 400:
                warnings.append("⚠️ CRITICAL: Extremely high blood glucose - seek immediate medical attention")
            
            if hba1c and hba1c > 14:
                warnings.append("⚠️ CRITICAL: Extremely high HbA1c - seek immediate medical attention")
            
        except Exception as e:
            logger.error(f"Error in combination validation: {str(e)}")
            warnings.append("Could not validate field combinations")
        
        return errors, warnings
    
    def validate_task_update(self, task_type: str, value: Any) -> Dict:
        """Validate health task update data"""
        try:
            from config import Config
            
            if task_type not in Config.DAILY_TASKS:
                return {
                    'valid': False,
                    'errors': [f"Invalid task type: {task_type}"]
                }
            
            task_config = Config.DAILY_TASKS[task_type]
            target = task_config['target']
            unit = task_config['unit']
            
            # Convert value to numeric
            try:
                numeric_value = float(value) if value is not None else 0
            except (ValueError, TypeError):
                return {
                    'valid': False,
                    'errors': [f"Task value must be numeric"]
                }
            
            # Validate range
            if numeric_value < 0:
                return {
                    'valid': False,
                    'errors': [f"Task value cannot be negative"]
                }
            
            # Check for unrealistic values
            max_reasonable = target * 3  # Allow up to 3x target
            warning = None
            
            if numeric_value > max_reasonable:
                warning = f"Value seems unusually high for {task_type} ({numeric_value} {unit})"
            
            return {
                'valid': True,
                'value': numeric_value,
                'completed': numeric_value >= target,
                'warning': warning
            }
            
        except Exception as e:
            logger.error(f"Error validating task update: {str(e)}")
            return {
                'valid': False,
                'errors': [f"Task validation failed: {str(e)}"]
            }
    
    def sanitize_user_input(self, text: str) -> str:
        """Sanitize user input for safety"""
        if not text:
            return ""
        
        # Remove potentially harmful characters
        sanitized = re.sub(r'[<>\"\'&]', '', str(text))
        
        # Limit length
        if len(sanitized) > 1000:
            sanitized = sanitized[:1000]
        
        # Remove excessive whitespace
        sanitized = re.sub(r'\s+', ' ', sanitized).strip()
        
        return sanitized
    
    def validate_email(self, email: str) -> bool:
        """Validate email format"""
        if not email:
            return False
        
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))
    
    def validate_date_range(self, start_date: str, end_date: str) -> Dict:
        """Validate date range"""
        try:
            from datetime import datetime
            
            start = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
            end = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
            
            if start > end:
                return {
                    'valid': False,
                    'errors': ['Start date must be before end date']
                }
            
            # Check reasonable range
            delta = end - start
            if delta.days > 365:
                return {
                    'valid': False,
                    'errors': ['Date range cannot exceed one year']
                }
            
            return {
                'valid': True,
                'start_date': start,
                'end_date': end,
                'days': delta.days
            }
            
        except ValueError as e:
            return {
                'valid': False,
                'errors': [f'Invalid date format: {str(e)}']
            }