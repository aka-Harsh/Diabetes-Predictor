from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import json
import logging
import numpy as np

logger = logging.getLogger(__name__)

def format_prediction_response(predictions: Dict, risk_analysis: Dict, recommendations: List[Dict]) -> Dict:
    """Format prediction response for frontend"""
    try:
        # Get the best/ensemble prediction
        main_prediction = predictions.get('ensemble', list(predictions.values())[0])
        
        response = {
            'timestamp': datetime.now().isoformat(),
            'primary_prediction': {
                'model': main_prediction.get('model', 'unknown'),
                'risk_level': main_prediction.get('risk_level', 'unknown'),
                'probability': main_prediction.get('probability', 0),
                'confidence_interval': main_prediction.get('confidence_interval', [0, 0]),
                'prediction_class': main_prediction.get('prediction', 0)
            },
            'all_predictions': predictions,
            'risk_analysis': risk_analysis,
            'recommendations': recommendations[:5],  # Top 5 recommendations
            'summary': generate_risk_summary(main_prediction, risk_analysis),
            'next_steps': get_immediate_next_steps(main_prediction.get('risk_level', 'low')),
            'interpretation': interpret_risk_level(main_prediction.get('risk_level', 'low'))
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Error formatting prediction response: {str(e)}")
        return {
            'error': 'Failed to format prediction response',
            'timestamp': datetime.now().isoformat()
        }

def generate_risk_summary(prediction: Dict, risk_analysis: Dict) -> str:
    """Generate human-readable risk summary"""
    try:
        risk_level = prediction.get('risk_level', 'unknown')
        probability = prediction.get('probability', 0)
        
        summaries = {
            'low': f"Your diabetes risk is currently low ({probability:.1%}). You're on the right track with your current health habits. Continue focusing on maintaining a healthy lifestyle to keep your risk low.",
            
            'moderate': f"Your diabetes risk is moderate ({probability:.1%}). This is a manageable level that can be improved with focused lifestyle changes. Consider this an opportunity to make positive health adjustments.",
            
            'high': f"Your diabetes risk is elevated ({probability:.1%}). While this may seem concerning, remember that risk can be significantly reduced through dedicated lifestyle modifications. Many people successfully lower their risk with proper guidance.",
            
            'very_high': f"Your diabetes risk is quite high ({probability:.1%}). This indicates the importance of taking immediate action, but don't be discouraged - significant risk reduction is possible with comprehensive lifestyle changes and medical support."
        }
        
        base_summary = summaries.get(risk_level, f"Your diabetes risk level is {risk_level}.")
        
        # Add key risk factors if available
        if risk_analysis and 'priority_interventions' in risk_analysis:
            priority_factors = risk_analysis['priority_interventions']
            if priority_factors:
                factors = [intervention['factor'] for intervention in priority_factors[:3]]
                base_summary += f" Key areas to focus on: {', '.join(factors)}."
        
        return base_summary
        
    except Exception as e:
        logger.error(f"Error generating risk summary: {str(e)}")
        return "Unable to generate risk summary at this time."

def interpret_risk_level(risk_level: str) -> Dict:
    """Provide detailed interpretation of risk level"""
    interpretations = {
        'low': {
            'meaning': 'Low Risk',
            'description': 'Your current lifestyle and health indicators suggest a low probability of developing diabetes.',
            'time_frame': 'This assessment reflects your risk over the next 5-10 years.',
            'action_required': 'Continue current healthy habits',
            'urgency': 'low',
            'color': 'green'
        },
        'moderate': {
            'meaning': 'Moderate Risk',
            'description': 'You have some risk factors that increase your likelihood of developing diabetes, but this is manageable.',
            'time_frame': 'Without intervention, risk may increase over the next 5-10 years.',
            'action_required': 'Implement targeted lifestyle improvements',
            'urgency': 'medium',
            'color': 'yellow'
        },
        'high': {
            'meaning': 'High Risk',
            'description': 'Multiple risk factors suggest a significant likelihood of developing diabetes without intervention.',
            'time_frame': 'Risk may materialize within 3-7 years without lifestyle changes.',
            'action_required': 'Begin comprehensive prevention program',
            'urgency': 'high',
            'color': 'orange'
        },
        'very_high': {
            'meaning': 'Very High Risk',
            'description': 'Your risk profile indicates an urgent need for intervention to prevent diabetes.',
            'time_frame': 'Risk may materialize within 1-5 years without immediate action.',
            'action_required': 'Seek immediate medical guidance and intensive lifestyle intervention',
            'urgency': 'critical',
            'color': 'red'
        }
    }
    
    return interpretations.get(risk_level, {
        'meaning': 'Unknown Risk',
        'description': 'Unable to interpret risk level.',
        'action_required': 'Consult healthcare provider',
        'urgency': 'medium',
        'color': 'gray'
    })

def get_immediate_next_steps(risk_level: str) -> List[Dict]:
    """Get immediate next steps based on risk level"""
    next_steps = {
        'low': [
            {
                'step': 'Continue Current Habits',
                'description': 'Maintain your current healthy lifestyle choices',
                'priority': 1,
                'timeframe': 'ongoing'
            },
            {
                'step': 'Annual Health Checkup',
                'description': 'Schedule yearly health screenings to monitor your status',
                'priority': 2,
                'timeframe': 'annually'
            },
            {
                'step': 'Track Health Metrics',
                'description': 'Use our health tracking tools to maintain awareness',
                'priority': 3,
                'timeframe': 'daily'
            }
        ],
        'moderate': [
            {
                'step': 'Start Prevention Program',
                'description': 'Begin a structured diabetes prevention program',
                'priority': 1,
                'timeframe': 'within 1 month'
            },
            {
                'step': 'Dietary Assessment',
                'description': 'Evaluate and improve your eating patterns',
                'priority': 2,
                'timeframe': 'within 2 weeks'
            },
            {
                'step': 'Increase Physical Activity',
                'description': 'Gradually increase exercise to meet recommended levels',
                'priority': 3,
                'timeframe': 'start immediately'
            }
        ],
        'high': [
            {
                'step': 'Medical Consultation',
                'description': 'Schedule appointment with healthcare provider within 2 weeks',
                'priority': 1,
                'timeframe': 'within 2 weeks'
            },
            {
                'step': 'Comprehensive Lifestyle Plan',
                'description': 'Develop detailed diet and exercise intervention plan',
                'priority': 2,
                'timeframe': 'within 1 week'
            },
            {
                'step': 'Blood Sugar Monitoring',
                'description': 'Begin regular monitoring of blood glucose levels',
                'priority': 3,
                'timeframe': 'start immediately'
            }
        ],
        'very_high': [
            {
                'step': 'Immediate Medical Attention',
                'description': 'Contact healthcare provider immediately for urgent assessment',
                'priority': 1,
                'timeframe': 'within 24-48 hours'
            },
            {
                'step': 'Intensive Prevention Program',
                'description': 'Enroll in intensive diabetes prevention program',
                'priority': 2,
                'timeframe': 'within 1 week'
            },
            {
                'step': 'Daily Health Monitoring',
                'description': 'Begin daily tracking of all health metrics',
                'priority': 3,
                'timeframe': 'start today'
            }
        ]
    }
    
    return next_steps.get(risk_level, [])

def calculate_health_score(user_id: str, db_manager) -> float:
    """Calculate comprehensive health score for user"""
    try:
        from datetime import date
        today = date.today()
        
        # Get recent health data
        recent_tasks = db_manager.get_daily_tasks(user_id, today)
        
        # Base score calculation
        total_possible = sum(task['points'] for task in recent_tasks)
        total_earned = sum(task['points_earned'] for task in recent_tasks if task['completed'])
        
        daily_score = (total_earned / total_possible * 100) if total_possible > 0 else 0
        
        # Get weekly trend
        weekly_scores = []
        for i in range(7):
            check_date = today - timedelta(days=i)
            day_tasks = db_manager.get_daily_tasks(user_id, check_date)
            if day_tasks:
                day_total = sum(task['points'] for task in day_tasks)
                day_earned = sum(task['points_earned'] for task in day_tasks if task['completed'])
                day_score = (day_earned / day_total * 100) if day_total > 0 else 0
                weekly_scores.append(day_score)
        
        # Calculate weighted score
        if weekly_scores:
            recent_avg = np.mean(weekly_scores[:3])  # Last 3 days
            weekly_avg = np.mean(weekly_scores)      # Full week
            
            # Weight recent performance higher
            health_score = (recent_avg * 0.6 + weekly_avg * 0.4)
        else:
            health_score = daily_score
        
        # Apply streak bonus
        current_streak = get_user_streak(user_id, db_manager)
        streak_bonus = min(current_streak * 2, 20)  # Max 20 points bonus
        
        final_score = min(health_score + streak_bonus, 100)
        return round(final_score, 1)
        
    except Exception as e:
        logger.error(f"Error calculating health score: {str(e)}")
        return 0.0

def get_user_streak(user_id: str, db_manager) -> int:
    """Get current streak for user"""
    try:
        # This is a simplified version - the full implementation would be in GamificationService
        today = datetime.now().date()
        streak = 0
        
        for i in range(365):  # Check up to a year
            check_date = today - timedelta(days=i)
            daily_tasks = db_manager.get_daily_tasks(user_id, check_date)
            
            # Check if any task was completed
            has_completion = any(task['completed'] for task in daily_tasks)
            
            if has_completion:
                streak += 1
            else:
                break
        
        return streak
        
    except Exception as e:
        logger.error(f"Error getting user streak: {str(e)}")
        return 0

def format_date_for_display(date_str: str, format_type: str = 'default') -> str:
    """Format date string for display"""
    try:
        if isinstance(date_str, str):
            dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        else:
            dt = date_str
        
        formats = {
            'default': '%B %d, %Y',
            'short': '%m/%d/%Y',
            'long': '%A, %B %d, %Y',
            'time': '%I:%M %p',
            'datetime': '%m/%d/%Y %I:%M %p',
            'iso': '%Y-%m-%d'
        }
        
        format_string = formats.get(format_type, formats['default'])
        return dt.strftime(format_string)
        
    except Exception as e:
        logger.error(f"Error formatting date: {str(e)}")
        return str(date_str)

def calculate_bmi_category(bmi: float) -> Dict:
    """Calculate BMI category and health implications"""
    try:
        if bmi < 18.5:
            category = 'Underweight'
            health_implications = 'May increase risk of malnutrition, osteoporosis, and anemia'
            color = 'blue'
        elif bmi < 25:
            category = 'Normal weight'
            health_implications = 'Associated with lowest risk of diabetes and cardiovascular disease'
            color = 'green'
        elif bmi < 30:
            category = 'Overweight'
            health_implications = 'Moderately increased risk of diabetes, heart disease, and stroke'
            color = 'yellow'
        elif bmi < 35:
            category = 'Obesity Class I'
            health_implications = 'Significantly increased risk of diabetes and cardiovascular complications'
            color = 'orange'
        elif bmi < 40:
            category = 'Obesity Class II'
            health_implications = 'High risk of serious health complications including diabetes'
            color = 'red'
        else:
            category = 'Obesity Class III'
            health_implications = 'Very high risk - immediate medical intervention recommended'
            color = 'red'
        
        return {
            'category': category,
            'health_implications': health_implications,
            'color': color,
            'bmi_value': round(bmi, 1)
        }
        
    except Exception as e:
        logger.error(f"Error calculating BMI category: {str(e)}")
        return {
            'category': 'Unknown',
            'health_implications': 'Unable to determine health implications',
            'color': 'gray',
            'bmi_value': bmi
        }

def generate_progress_insights(user_data: Dict, historical_data: List[Dict]) -> List[Dict]:
    """Generate insights from user progress data"""
    try:
        insights = []
        
        if len(historical_data) < 2:
            return [{
                'type': 'welcome',
                'title': 'Welcome to Your Health Journey',
                'message': 'Start tracking your health metrics to see personalized insights and progress over time.',
                'priority': 'low'
            }]
        
        # Analyze trend in health scores
        recent_scores = [data.get('health_score', 0) for data in historical_data[-7:]]
        older_scores = [data.get('health_score', 0) for data in historical_data[-14:-7]]
        
        if len(recent_scores) >= 3 and len(older_scores) >= 3:
            recent_avg = np.mean(recent_scores)
            older_avg = np.mean(older_scores)
            improvement = recent_avg - older_avg
            
            if improvement > 10:
                insights.append({
                    'type': 'improvement',
                    'title': 'Great Progress!',
                    'message': f'Your health score has improved by {improvement:.1f} points over the past week. Keep up the excellent work!',
                    'priority': 'high'
                })
            elif improvement < -10:
                insights.append({
                    'type': 'attention_needed',
                    'title': 'Health Score Declining',
                    'message': f'Your health score has decreased by {abs(improvement):.1f} points. Consider focusing on your daily health tasks.',
                    'priority': 'medium'
                })
        
        # Analyze consistency
        completion_rates = [data.get('completion_rate', 0) for data in historical_data[-7:]]
        if completion_rates:
            consistency = np.std(completion_rates)
            if consistency < 0.2:  # Very consistent
                insights.append({
                    'type': 'consistency',
                    'title': 'Excellent Consistency',
                    'message': 'You\'ve been very consistent with your health tasks. Consistency is key to long-term health improvements.',
                    'priority': 'medium'
                })
        
        return insights
        
    except Exception as e:
        logger.error(f"Error generating progress insights: {str(e)}")
        return []

def calculate_risk_change(current_prediction: Dict, previous_prediction: Dict) -> Dict:
    """Calculate change in risk between predictions"""
    try:
        current_prob = current_prediction.get('probability', 0)
        previous_prob = previous_prediction.get('probability', 0)
        
        change = current_prob - previous_prob
        percent_change = (change / previous_prob * 100) if previous_prob > 0 else 0
        
        if abs(change) < 0.05:
            trend = 'stable'
            significance = 'minimal'
        elif change > 0:
            trend = 'increasing'
            significance = 'moderate' if change < 0.15 else 'significant'
        else:
            trend = 'decreasing'
            significance = 'moderate' if abs(change) < 0.15 else 'significant'
        
        return {
            'absolute_change': round(change, 3),
            'percent_change': round(percent_change, 1),
            'trend': trend,
            'significance': significance,
            'current_risk': current_prob,
            'previous_risk': previous_prob
        }
        
    except Exception as e:
        logger.error(f"Error calculating risk change: {str(e)}")
        return {
            'absolute_change': 0,
            'percent_change': 0,
            'trend': 'unknown',
            'significance': 'unknown'
        }

def generate_export_filename(user_id: str, data_type: str = 'health_data') -> str:
    """Generate filename for data export"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"{data_type}_{user_id[:8]}_{timestamp}.json"

def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe storage"""
    import re
    # Remove or replace unsafe characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Limit length
    if len(sanitized) > 255:
        name, ext = sanitized.rsplit('.', 1) if '.' in sanitized else (sanitized, '')
        sanitized = name[:250] + ('.' + ext if ext else '')
    return sanitized

def format_percentage(value: float, decimals: int = 1) -> str:
    """Format percentage for display"""
    try:
        return f"{value:.{decimals}f}%"
    except:
        return "0.0%"

def get_risk_color(risk_level: str) -> str:
    """Get color code for risk level"""
    colors = {
        'low': '#10B981',      # Green
        'moderate': '#F59E0B', # Yellow
        'high': '#F97316',     # Orange
        'very_high': '#EF4444' # Red
    }
    return colors.get(risk_level, '#6B7280')  # Gray default

def chunk_list(lst: List, chunk_size: int) -> List[List]:
    """Split list into chunks of specified size"""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def safe_divide(numerator: float, denominator: float, default: float = 0) -> float:
    """Safely divide two numbers, returning default if denominator is zero"""
    try:
        return numerator / denominator if denominator != 0 else default
    except (TypeError, ZeroDivisionError):
        return default

def generate_health_tip() -> Dict:
    """Generate a random health tip"""
    tips = [
        {
            'title': 'Stay Hydrated',
            'content': 'Drinking water helps regulate blood sugar levels and supports overall metabolism.',
            'category': 'nutrition'
        },
        {
            'title': 'Walk After Meals',
            'content': 'A 10-15 minute walk after eating can help lower blood sugar spikes.',
            'category': 'exercise'
        },
        {
            'title': 'Get Quality Sleep',
            'content': 'Poor sleep can affect hormone levels and increase diabetes risk. Aim for 7-9 hours nightly.',
            'category': 'lifestyle'
        },
        {
            'title': 'Manage Stress',
            'content': 'Chronic stress can raise blood sugar levels. Try deep breathing, meditation, or yoga.',
            'category': 'wellness'
        },
        {
            'title': 'Choose Whole Grains',
            'content': 'Whole grains provide steady energy and better blood sugar control than refined grains.',
            'category': 'nutrition'
        },
        {
            'title': 'Track Your Progress',
            'content': 'Regular monitoring helps you understand patterns and make informed health decisions.',
            'category': 'monitoring'
        }
    ]
    
    import random
    return random.choice(tips)

def validate_json_structure(data: Any, required_fields: List[str]) -> Tuple[bool, List[str]]:
    """Validate JSON data structure"""
    errors = []
    
    if not isinstance(data, dict):
        errors.append("Data must be a JSON object")
        return False, errors
    
    for field in required_fields:
        if field not in data:
            errors.append(f"Missing required field: {field}")
    
    return len(errors) == 0, errors

def calculate_age_from_birthdate(birthdate: str) -> int:
    """Calculate age from birthdate string"""
    try:
        birth = datetime.strptime(birthdate, '%Y-%m-%d')
        today = datetime.now()
        age = today.year - birth.year - ((today.month, today.day) < (birth.month, birth.day))
        return max(0, age)
    except:
        return 0

def format_currency(amount: float, currency: str = 'USD') -> str:
    """Format amount as currency"""
    try:
        if currency == 'USD':
            return f"${amount:,.2f}"
        else:
            return f"{amount:,.2f} {currency}"
    except:
        return f"0.00 {currency}"

def generate_unique_id() -> str:
    """Generate unique identifier"""
    import uuid
    return str(uuid.uuid4())

def is_weekend(date_obj: datetime) -> bool:
    """Check if date is weekend"""
    return date_obj.weekday() >= 5  # Saturday = 5, Sunday = 6

def calculate_percentile(value: float, data_list: List[float]) -> float:
    """Calculate percentile of value in data list"""
    try:
        if not data_list:
            return 0.0
        
        data_sorted = sorted(data_list)
        n = len(data_sorted)
        
        if value <= data_sorted[0]:
            return 0.0
        if value >= data_sorted[-1]:
            return 100.0
        
        # Find position
        for i, v in enumerate(data_sorted):
            if value <= v:
                return (i / n) * 100
        
        return 100.0
    except:
        return 0.0

def clean_text(text: str) -> str:
    """Clean and normalize text"""
    if not text:
        return ""
    
    # Remove extra whitespace
    cleaned = ' '.join(text.split())
    
    # Remove special characters but keep basic punctuation
    import re
    cleaned = re.sub(r'[^\w\s\-\.\,\!\?\:\;]', '', cleaned)
    
    return cleaned.strip()

def calculate_compound_interest(principal: float, rate: float, time: float, 
                              compound_frequency: int = 12) -> float:
    """Calculate compound interest"""
    try:
        if rate <= 0 or time <= 0:
            return principal
        
        amount = principal * (1 + rate / compound_frequency) ** (compound_frequency * time)
        return round(amount, 2)
    except:
        return principal

def time_ago(dt: datetime) -> str:
    """Get human-readable time difference"""
    try:
        now = datetime.now()
        diff = now - dt
        
        if diff.days > 0:
            return f"{diff.days} day{'s' if diff.days > 1 else ''} ago"
        elif diff.seconds > 3600:
            hours = diff.seconds // 3600
            return f"{hours} hour{'s' if hours > 1 else ''} ago"
        elif diff.seconds > 60:
            minutes = diff.seconds // 60
            return f"{minutes} minute{'s' if minutes > 1 else ''} ago"
        else:
            return "Just now"
    except:
        return "Unknown"

def merge_dictionaries(*dicts: Dict) -> Dict:
    """Merge multiple dictionaries"""
    result = {}
    for d in dicts:
        if isinstance(d, dict):
            result.update(d)
    return result

def flatten_list(nested_list: List[Any]) -> List[Any]:
    """Flatten nested list"""
    result = []
    for item in nested_list:
        if isinstance(item, list):
            result.extend(flatten_list(item))
        else:
            result.append(item)
    return result