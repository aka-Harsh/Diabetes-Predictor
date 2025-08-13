from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
import os
import json
from datetime import datetime, timedelta
import logging

from config import Config
from services.prediction_service import PredictionService
from services.health_service import HealthService
from services.gamification_service import GamificationService
from services.ollama_service import OllamaService
from utils.database import DatabaseManager
from utils.validators import InputValidator
from utils.helpers import format_prediction_response, calculate_health_score

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(Config)
CORS(app)

# Initialize services
db_manager = DatabaseManager(app.config['DATABASE_URL'])
prediction_service = PredictionService()
health_service = HealthService()
gamification_service = GamificationService()
ollama_service = OllamaService()
validator = InputValidator()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route('/')
def index():
    """Main dashboard route"""
    try:
        # Get user session or create new one
        if 'user_id' not in session:
            session['user_id'] = db_manager.create_user_session()
        
        user_id = session['user_id']
        
        # Get user statistics
        stats = {
            'total_predictions': db_manager.get_user_prediction_count(user_id),
            'current_streak': gamification_service.get_current_streak(user_id),
            'health_score': calculate_health_score(user_id, db_manager),
            'recent_predictions': db_manager.get_recent_predictions(user_id, limit=5)
        }
        
        return render_template('index.html', stats=stats)
    except Exception as e:
        logger.error(f"Error in index route: {str(e)}")
        return render_template('index.html', stats={})

@app.route('/prediction-form')
def prediction_form():
    """Multi-step prediction form"""
    return render_template('prediction-form.html')

@app.route('/api/predict', methods=['POST'])
def predict_diabetes():
    """Main prediction endpoint"""
    try:
        data = request.json
        
        # Validate input data
        validation_result = validator.validate_prediction_input(data)
        if not validation_result['valid']:
            return jsonify({'error': validation_result['errors']}), 400
        
        # Get predictions from multiple models
        model_type = data.get('model_type', 'random_forest')
        predictions = prediction_service.predict_multiple_models(data)
        
        # Calculate risk factors and insights
        risk_analysis = health_service.analyze_risk_factors(data)
        recommendations = health_service.get_personalized_recommendations(data, predictions)
        
        # Store prediction in database
        user_id = session.get('user_id')
        if user_id:
            db_manager.store_prediction(user_id, data, predictions, risk_analysis)
        
        # Format response
        response = format_prediction_response(predictions, risk_analysis, recommendations)
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        return jsonify({'error': 'Prediction failed. Please try again.'}), 500

@app.route('/api/health-tasks', methods=['GET', 'POST'])
def health_tasks():
    """Health task tracking endpoints"""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'User session required'}), 401
    
    if request.method == 'GET':
        # Get today's tasks
        today = datetime.now().date()
        try:
            tasks = gamification_service.get_daily_tasks(user_id, today)
            logger.info(f"Retrieved {len(tasks.get('tasks', {}))} tasks for user {user_id}")
            return jsonify(tasks)
        except Exception as e:
            logger.error(f"Error getting health tasks: {str(e)}")
            # Return default task structure
            from config import Config
            default_tasks = {}
            for task_name, task_config in Config.DAILY_TASKS.items():
                default_tasks[task_name] = {
                    'name': task_name,
                    'display_name': task_name.replace('_', ' ').title(),
                    'target': task_config['target'],
                    'unit': task_config['unit'],
                    'points': task_config['points'],
                    'completed': False,
                    'current_value': 0,
                    'points_earned': 0,
                    'icon': gamification_service._get_task_icon(task_name),
                    'category': gamification_service._get_task_category(task_name)
                }
            
            return jsonify({
                'date': today.isoformat(),
                'tasks': default_tasks,
                'total_possible_points': sum(config['points'] for config in Config.DAILY_TASKS.values()),
                'earned_points': 0,
                'completion_rate': 0
            })
    
    elif request.method == 'POST':
        # Update task completion
        data = request.json
        task_type = data.get('task_type')
        completed = data.get('completed', False)
        value = data.get('value', 0)
        
        try:
            result = gamification_service.update_task_completion(
                user_id, task_type, completed, value
            )
            
            # Calculate new health score
            health_score = calculate_health_score(user_id, db_manager)
            result['health_score'] = health_score
            
            logger.info(f"Updated task {task_type} for user {user_id}: {result}")
            return jsonify(result)
        except Exception as e:
            logger.error(f"Error updating task completion: {str(e)}")
            return jsonify({'error': 'Failed to update task', 'success': False}), 500

@app.route('/api/gamification/stats')
def gamification_stats():
    """Get gamification statistics"""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'User session required'}), 401
    
    stats = gamification_service.get_user_stats(user_id)
    return jsonify(stats)

@app.route('/api/charts/risk-factors')
def risk_factors_chart():
    """Get risk factors data for charts"""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'User session required'}), 401
    
    # Get latest prediction data
    latest_prediction = db_manager.get_latest_prediction(user_id)
    if not latest_prediction:
        return jsonify({'error': 'No prediction data available'}), 404
    
    chart_data = health_service.generate_risk_chart_data(latest_prediction)
    return jsonify(chart_data)

@app.route('/api/charts/trends')
def health_trends_chart():
    """Get health trends data for charts"""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'User session required'}), 401
    
    days = request.args.get('days', 30, type=int)
    
    try:
        trends_data = db_manager.get_health_trends(user_id, days)
        
        # If no data, return sample data for demonstration
        if not trends_data.get('daily_completion') and not trends_data.get('risk_trend'):
            sample_data = {
                'daily_completion': [
                    {'date': (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d'), 
                     'completion_rate': 0.6 + (i % 3) * 0.1, 
                     'points': 45 + (i % 4) * 10} 
                    for i in range(7, 0, -1)
                ],
                'risk_trend': []
            }
            return jsonify(sample_data)
        
        return jsonify(trends_data)
    except Exception as e:
        logger.error(f"Error getting health trends: {str(e)}")
        # Return sample data on error
        sample_data = {
            'daily_completion': [
                {'date': (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d'), 
                 'completion_rate': 0.5 + (i % 3) * 0.15, 
                 'points': 40 + (i % 4) * 12} 
                for i in range(7, 0, -1)
            ],
            'risk_trend': []
        }
        return jsonify(sample_data)

@app.route('/api/ai-chat', methods=['POST'])
def ai_chat():
    """AI chatbot endpoint using Ollama"""
    try:
        data = request.json
        message = data.get('message', '')
        user_id = session.get('user_id')
        
        # Get user context for personalized responses
        context = {}
        if user_id:
            context = db_manager.get_user_health_context(user_id)
        
        # Get AI response
        response = ollama_service.get_health_guidance(message, context)
        
        return jsonify({'response': response})
        
    except Exception as e:
        logger.error(f"Error in AI chat: {str(e)}")
        return jsonify({'error': 'AI service temporarily unavailable'}), 500

@app.route('/api/scenarios', methods=['POST'])
def scenario_analysis():
    """What-if scenario analysis"""
    try:
        data = request.json
        base_data = data.get('base_data', {})
        modifications = data.get('modifications', {})
        
        # Create modified data for scenario
        scenario_data = base_data.copy()
        scenario_data.update(modifications)
        
        # Get predictions for scenario
        predictions = prediction_service.predict_multiple_models(scenario_data)
        
        # Compare with baseline if available
        baseline_predictions = None
        if base_data:
            baseline_predictions = prediction_service.predict_multiple_models(base_data)
        
        response = {
            'scenario_predictions': predictions,
            'baseline_predictions': baseline_predictions,
            'risk_reduction': None
        }
        
        if baseline_predictions:
            response['risk_reduction'] = {
                model: predictions[model]['probability'] - baseline_predictions[model]['probability']
                for model in predictions.keys()
            }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in scenario analysis: {str(e)}")
        return jsonify({'error': 'Scenario analysis failed'}), 500

@app.route('/api/population-stats')
def population_statistics():
    """Get population-level statistics for comparison"""
    try:
        stats = db_manager.get_population_statistics()
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Error getting population stats: {str(e)}")
        return jsonify({'error': 'Statistics unavailable'}), 500

@app.route('/api/export-data')
def export_user_data():
    """Export user health data"""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'User session required'}), 401
    
    try:
        data = db_manager.export_user_data(user_id)
        return jsonify(data)
    except Exception as e:
        logger.error(f"Error exporting data: {str(e)}")
        return jsonify({'error': 'Export failed'}), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return render_template('index.html'), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal error: {str(error)}")
    return render_template('index.html'), 500

if __name__ == '__main__':
    # Initialize database on startup
    try:
        db_manager.initialize_database()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {str(e)}")
    
    # Run the application
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    
    app.run(host='0.0.0.0', port=port, debug=debug)