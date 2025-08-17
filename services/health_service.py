import numpy as np
from typing import Dict, List, Tuple, Any
import logging
from config import Config

logger = logging.getLogger(__name__)

class HealthService:
    def __init__(self):
        self.risk_factors = {
            'age': {'weight': 0.15, 'threshold': 45},
            'bmi': {'weight': 0.20, 'threshold': 25},
            'blood_glucose': {'weight': 0.25, 'threshold': 126},
            'hba1c': {'weight': 0.25, 'threshold': 6.5},
            'hypertension': {'weight': 0.10, 'threshold': 1},
            'heart_disease': {'weight': 0.15, 'threshold': 1},
            'smoking': {'weight': 0.10, 'threshold': 1}
        }
    
    def analyze_risk_factors(self, user_data: Dict) -> Dict:
        """Analyze individual risk factors and their contributions"""
        try:
            risk_analysis = {
                'individual_factors': {},
                'risk_score': 0,
                'modifiable_factors': [],
                'non_modifiable_factors': [],
                'priority_interventions': []
            }
            
            total_risk = 0
            
            # Analyze each risk factor
            for factor, config in self.risk_factors.items():
                value = self._get_factor_value(user_data, factor)
                normalized_risk = self._calculate_factor_risk(factor, value, config)
                weighted_risk = normalized_risk * config['weight']
                
                risk_analysis['individual_factors'][factor] = {
                    'value': value,
                    'risk_score': normalized_risk,
                    'weighted_contribution': weighted_risk,
                    'modifiable': self._is_modifiable(factor),
                    'severity': self._get_severity_level(normalized_risk)
                }
                
                total_risk += weighted_risk
                
                # Categorize factors
                if self._is_modifiable(factor):
                    risk_analysis['modifiable_factors'].append(factor)
                else:
                    risk_analysis['non_modifiable_factors'].append(factor)
                
                # Add to priority interventions if high risk and modifiable
                if normalized_risk > 0.6 and self._is_modifiable(factor):
                    risk_analysis['priority_interventions'].append({
                        'factor': factor,
                        'current_risk': normalized_risk,
                        'potential_impact': config['weight']
                    })
            
            risk_analysis['risk_score'] = total_risk
            risk_analysis['overall_risk_level'] = self._get_overall_risk_level(total_risk)
            
            return risk_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing risk factors: {str(e)}")
            return {'error': str(e)}
    
    def _get_factor_value(self, data: Dict, factor: str) -> float:
        """Extract factor value from user data"""
        mapping = {
            'age': data.get('age', 40),
            'bmi': data.get('bmi', 25),
            'blood_glucose': data.get('blood_glucose_level', 100),
            'hba1c': data.get('hba1c_level', 5.5),
            'hypertension': 1 if data.get('hypertension') else 0,
            'heart_disease': 1 if data.get('heart_disease') else 0,
            'smoking': 1 if data.get('smoking_history') == 'current' else 0
        }
        return mapping.get(factor, 0)
    
    def _calculate_factor_risk(self, factor: str, value: float, config: Dict) -> float:
        """Calculate normalized risk for a specific factor"""
        threshold = config['threshold']
        
        if factor in ['age', 'bmi', 'blood_glucose', 'hba1c']:
            # Continuous variables
            if value <= threshold:
                return 0.0
            else:
                # Scale based on how much above threshold
                if factor == 'age':
                    return min(1.0, (value - threshold) / 30)  # Max risk at 75
                elif factor == 'bmi':
                    return min(1.0, (value - threshold) / 15)  # Max risk at BMI 40
                elif factor == 'blood_glucose':
                    return min(1.0, (value - threshold) / 100) # Max risk at 226
                elif factor == 'hba1c':
                    return min(1.0, (value - threshold) / 4)   # Max risk at HbA1c 10.5
        else:
            # Binary variables
            return float(value)
    
    def _is_modifiable(self, factor: str) -> bool:
        """Check if risk factor is modifiable"""
        modifiable_factors = ['bmi', 'blood_glucose', 'hba1c', 'smoking', 'hypertension']
        return factor in modifiable_factors
    
    def _get_severity_level(self, risk_score: float) -> str:
        """Get severity level for individual factor"""
        if risk_score < 0.3:
            return 'low'
        elif risk_score < 0.6:
            return 'moderate'
        elif risk_score < 0.8:
            return 'high'
        else:
            return 'very_high'
    
    def _get_overall_risk_level(self, total_risk: float) -> str:
        """Get overall risk level"""
        if total_risk < 0.25:
            return 'low'
        elif total_risk < 0.50:
            return 'moderate'
        elif total_risk < 0.75:
            return 'high'
        else:
            return 'very_high'
    
    def get_personalized_recommendations(self, user_data: Dict, predictions: Dict) -> List[Dict]:
        """Generate personalized health recommendations"""
        try:
            recommendations = []
            
            # Get risk analysis
            risk_analysis = self.analyze_risk_factors(user_data)
            
            # Generate recommendations based on priority interventions
            for intervention in risk_analysis.get('priority_interventions', []):
                factor = intervention['factor']
                recommendations.extend(self._get_factor_recommendations(factor, user_data))
            
            # Add general recommendations based on overall risk
            overall_risk = risk_analysis.get('overall_risk_level', 'low')
            recommendations.extend(self._get_general_recommendations(overall_risk))
            
            # Sort by priority and remove duplicates
            unique_recommendations = []
            seen_titles = set()
            
            for rec in recommendations:
                if rec['title'] not in seen_titles:
                    unique_recommendations.append(rec)
                    seen_titles.add(rec['title'])
            
            # Sort by priority (higher priority first)
            unique_recommendations.sort(key=lambda x: x.get('priority', 5), reverse=True)
            
            return unique_recommendations[:10]  # Return top 10
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            return []
    
    def _get_factor_recommendations(self, factor: str, user_data: Dict) -> List[Dict]:
        """Get recommendations for specific risk factor"""
        recommendations = {
            'bmi': [
                {
                    'title': 'Weight Management',
                    'description': 'Aim for gradual weight loss through balanced diet and regular exercise',
                    'action_items': [
                        'Create a caloric deficit of 500-750 calories per day',
                        'Focus on whole foods and portion control',
                        'Incorporate strength training 2-3 times per week'
                    ],
                    'priority': 9,
                    'category': 'nutrition'
                }
            ],
            'blood_glucose': [
                {
                    'title': 'Blood Sugar Control',
                    'description': 'Manage blood glucose through diet, exercise, and monitoring',
                    'action_items': [
                        'Monitor blood glucose regularly',
                        'Choose low glycemic index foods',
                        'Exercise after meals to help glucose uptake'
                    ],
                    'priority': 10,
                    'category': 'medical'
                }
            ],
            'hba1c': [
                {
                    'title': 'Long-term Glucose Management',
                    'description': 'Improve HbA1c through consistent lifestyle modifications',
                    'action_items': [
                        'Maintain consistent meal timing',
                        'Work with healthcare provider on medication optimization',
                        'Track daily glucose patterns'
                    ],
                    'priority': 10,
                    'category': 'medical'
                }
            ],
            'smoking': [
                {
                    'title': 'Smoking Cessation',
                    'description': 'Quit smoking to dramatically reduce diabetes and cardiovascular risk',
                    'action_items': [
                        'Consult healthcare provider about cessation aids',
                        'Join a smoking cessation program',
                        'Replace smoking triggers with healthy activities'
                    ],
                    'priority': 10,
                    'category': 'lifestyle'
                }
            ],
            'hypertension': [
                {
                    'title': 'Blood Pressure Management',
                    'description': 'Control hypertension through lifestyle and medical management',
                    'action_items': [
                        'Reduce sodium intake to less than 2300mg daily',
                        'Engage in regular aerobic exercise',
                        'Take prescribed medications consistently'
                    ],
                    'priority': 8,
                    'category': 'medical'
                }
            ]
        }
        
        return recommendations.get(factor, [])
    
    def _get_general_recommendations(self, risk_level: str) -> List[Dict]:
        """Get general recommendations based on overall risk level"""
        base_recommendations = [
            {
                'title': 'Regular Health Checkups',
                'description': 'Schedule regular medical appointments for monitoring',
                'action_items': [
                    'Annual comprehensive physical exam',
                    'Regular blood work including glucose and HbA1c',
                    'Blood pressure monitoring'
                ],
                'priority': 7,
                'category': 'medical'
            },
            {
                'title': 'Healthy Diet Pattern',
                'description': 'Adopt a balanced, diabetes-friendly eating pattern',
                'action_items': [
                    'Fill half your plate with non-starchy vegetables',
                    'Choose whole grains over refined carbohydrates',
                    'Include lean protein sources'
                ],
                'priority': 8,
                'category': 'nutrition'
            },
            {
                'title': 'Regular Physical Activity',
                'description': 'Incorporate consistent exercise into daily routine',
                'action_items': [
                    'Aim for 150 minutes of moderate aerobic activity weekly',
                    'Include strength training exercises twice weekly',
                    'Take regular walks after meals'
                ],
                'priority': 8,
                'category': 'exercise'
            }
        ]
        
        if risk_level in ['high', 'very_high']:
            base_recommendations.extend([
                {
                    'title': 'Diabetes Prevention Program',
                    'description': 'Consider enrolling in a structured prevention program',
                    'action_items': [
                        'Look into CDC-recognized diabetes prevention programs',
                        'Work with a registered dietitian',
                        'Consider continuous glucose monitoring'
                    ],
                    'priority': 9,
                    'category': 'medical'
                },
                {
                    'title': 'Stress Management',
                    'description': 'Implement stress reduction techniques',
                    'action_items': [
                        'Practice mindfulness or meditation daily',
                        'Ensure adequate sleep (7-9 hours)',
                        'Consider counseling if needed'
                    ],
                    'priority': 6,
                    'category': 'lifestyle'
                }
            ])
        
        return base_recommendations
    
    def generate_risk_chart_data(self, prediction_data: Dict) -> Dict:
        """Generate data for risk factor visualization"""
        try:
            input_data = prediction_data.get('input_data', {})
            risk_analysis = self.analyze_risk_factors(input_data)
            
            # Prepare radar chart data
            factors = []
            values = []
            max_values = []
            
            for factor, analysis in risk_analysis.get('individual_factors', {}).items():
                factors.append(self._get_factor_display_name(factor))
                values.append(analysis['risk_score'])
                max_values.append(1.0)  # All factors normalized to 0-1
            
            # Prepare bar chart data for factor contributions
            contributions = []
            for factor, analysis in risk_analysis.get('individual_factors', {}).items():
                contributions.append({
                    'factor': self._get_factor_display_name(factor),
                    'contribution': analysis['weighted_contribution'],
                    'modifiable': analysis['modifiable']
                })
            
            # Sort by contribution
            contributions.sort(key=lambda x: x['contribution'], reverse=True)
            
            return {
                'radar_chart': {
                    'factors': factors,
                    'values': values,
                    'max_values': max_values
                },
                'contribution_chart': contributions,
                'overall_risk': risk_analysis.get('risk_score', 0),
                'risk_level': risk_analysis.get('overall_risk_level', 'low')
            }
            
        except Exception as e:
            logger.error(f"Error generating chart data: {str(e)}")
            return {}
    
    def _get_factor_display_name(self, factor: str) -> str:
        """Get user-friendly display name for factor"""
        display_names = {
            'age': 'Age',
            'bmi': 'BMI',
            'blood_glucose': 'Blood Glucose',
            'hba1c': 'HbA1c',
            'hypertension': 'High Blood Pressure',
            'heart_disease': 'Heart Disease',
            'smoking': 'Smoking Status'
        }
        return display_names.get(factor, factor.title())
    
    def calculate_lifestyle_impact(self, current_data: Dict, modifications: Dict) -> Dict:
        """Calculate potential impact of lifestyle modifications"""
        try:
            # Current risk
            current_risk = self.analyze_risk_factors(current_data)
            
            # Modified data
            modified_data = current_data.copy()
            modified_data.update(modifications)
            modified_risk = self.analyze_risk_factors(modified_data)
            
            # Calculate improvements
            improvements = {}
            for factor in current_risk.get('individual_factors', {}):
                current_factor_risk = current_risk['individual_factors'][factor]['weighted_contribution']
                modified_factor_risk = modified_risk['individual_factors'][factor]['weighted_contribution']
                
                improvements[factor] = {
                    'current_risk': current_factor_risk,
                    'modified_risk': modified_factor_risk,
                    'improvement': current_factor_risk - modified_factor_risk,
                    'percent_improvement': ((current_factor_risk - modified_factor_risk) / current_factor_risk * 100) if current_factor_risk > 0 else 0
                }
            
            total_improvement = current_risk['risk_score'] - modified_risk['risk_score']
            
            return {
                'current_risk_score': current_risk['risk_score'],
                'modified_risk_score': modified_risk['risk_score'],
                'total_improvement': total_improvement,
                'percent_improvement': (total_improvement / current_risk['risk_score'] * 100) if current_risk['risk_score'] > 0 else 0,
                'factor_improvements': improvements,
                'risk_level_change': {
                    'from': current_risk['overall_risk_level'],
                    'to': modified_risk['overall_risk_level']
                }
            }
            
        except Exception as e:
            logger.error(f"Error calculating lifestyle impact: {str(e)}")
            return {}
    
    def get_health_insights(self, user_data: Dict, prediction_history: List[Dict]) -> List[Dict]:
        """Generate health insights based on user data and history"""
        try:
            insights = []
            
            # Current risk analysis
            current_risk = self.analyze_risk_factors(user_data)
            
            # Risk trend analysis
            if len(prediction_history) > 1:
                trend_insight = self._analyze_risk_trend(prediction_history)
                if trend_insight:
                    insights.append(trend_insight)
            
            # Modifiable risk factors insight
            modifiable_factors = current_risk.get('modifiable_factors', [])
            if modifiable_factors:
                insights.append({
                    'type': 'modifiable_factors',
                    'title': 'Controllable Risk Factors',
                    'description': f'You have {len(modifiable_factors)} risk factors that can be improved through lifestyle changes.',
                    'factors': modifiable_factors,
                    'priority': 'high'
                })
            
            # Positive reinforcement
            low_risk_factors = []
            for factor, analysis in current_risk.get('individual_factors', {}).items():
                if analysis['risk_score'] < 0.3:
                    low_risk_factors.append(factor)
            
            if low_risk_factors:
                insights.append({
                    'type': 'positive_reinforcement',
                    'title': 'Good Health Indicators',
                    'description': f'You have {len(low_risk_factors)} risk factors in the healthy range. Keep up the good work!',
                    'factors': low_risk_factors,
                    'priority': 'medium'
                })
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating health insights: {str(e)}")
            return []
    
    def _analyze_risk_trend(self, prediction_history: List[Dict]) -> Dict:
        """Analyze risk trend from prediction history"""
        try:
            if len(prediction_history) < 2:
                return None
            
            # Get risk scores over time
            risk_scores = []
            for prediction in prediction_history[-5:]:  # Last 5 predictions
                if 'risk_analysis' in prediction and prediction['risk_analysis']:
                    risk_scores.append(prediction['risk_analysis'].get('risk_score', 0))
            
            if len(risk_scores) < 2:
                return None
            
            # Calculate trend
            recent_avg = np.mean(risk_scores[-2:])
            older_avg = np.mean(risk_scores[:-2]) if len(risk_scores) > 2 else risk_scores[0]
            
            trend = recent_avg - older_avg
            
            if abs(trend) < 0.05:
                trend_description = "stable"
                priority = "low"
            elif trend > 0:
                trend_description = "increasing"
                priority = "high"
            else:
                trend_description = "improving"
                priority = "medium"
            
            return {
                'type': 'risk_trend',
                'title': f'Risk Trend: {trend_description.title()}',
                'description': f'Your diabetes risk has been {trend_description} over recent assessments.',
                'trend_value': trend,
                'priority': priority
            }
            
        except Exception as e:
            logger.error(f"Error analyzing risk trend: {str(e)}")
            return None