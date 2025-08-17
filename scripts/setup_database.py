#!/usr/bin/env python3
"""
Enhanced Diabetes Predictor - Database Setup Script
"""

import os
import sys
import logging
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from utils.database import DatabaseManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_database():
    """Initialize database and create all tables"""
    logger.info("Setting up Enhanced Diabetes Predictor database...")
    
    try:
        # Initialize database manager
        db_manager = DatabaseManager(Config.DATABASE_URL)
        
        logger.info("Database tables created successfully!")
        
        # Create some sample data for testing
        create_sample_data(db_manager)
        
        return True
        
    except Exception as e:
        logger.error(f"Database setup failed: {str(e)}")
        return False

def create_sample_data(db_manager):
    """Create sample data for testing"""
    logger.info("Creating sample data...")
    
    try:
        # Create a sample user session
        user_id = db_manager.create_user_session()
        logger.info(f"Created sample user: {user_id}")
        
        # Store sample prediction
        sample_input = {
            'age': 45,
            'gender': 'Male',
            'bmi': 28.5,
            'hypertension': True,
            'heart_disease': False,
            'smoking_history': 'former',
            'hba1c_level': 6.1,
            'blood_glucose_level': 125
        }
        
        sample_predictions = {
            'random_forest': {
                'model': 'random_forest',
                'prediction': 0,
                'probability': 0.35,
                'risk_level': 'moderate',
                'confidence_interval': [0.25, 0.45]
            },
            'xgboost': {
                'model': 'xgboost',
                'prediction': 0,
                'probability': 0.32,
                'risk_level': 'moderate',
                'confidence_interval': [0.22, 0.42]
            },
            'logistic_regression': {
                'model': 'logistic_regression',
                'prediction': 0,
                'probability': 0.38,
                'risk_level': 'moderate',
                'confidence_interval': [0.28, 0.48]
            }
        }
        
        sample_risk_analysis = {
            'overall_risk_level': 'moderate',
            'risk_score': 0.35,
            'individual_factors': {
                'age': {'risk_score': 0.3, 'modifiable': False},
                'bmi': {'risk_score': 0.6, 'modifiable': True},
                'hba1c': {'risk_score': 0.7, 'modifiable': True}
            },
            'modifiable_factors': ['bmi', 'hba1c'],
            'priority_interventions': [
                {'factor': 'bmi', 'current_risk': 0.6, 'potential_impact': 0.2},
                {'factor': 'hba1c', 'current_risk': 0.7, 'potential_impact': 0.25}
            ]
        }
        
        db_manager.store_prediction(user_id, sample_input, sample_predictions, sample_risk_analysis)
        logger.info("Sample prediction stored successfully")
        
        # Store sample health tasks
        from config import Config
        today = datetime.now().date()
        
        for task_name, task_config in Config.DAILY_TASKS.items():
            completed = task_name in ['water_intake', 'sleep_hours']  # Mark some as completed
            value = task_config['target'] if completed else task_config['target'] * 0.7
            points = task_config['points'] if completed else int(task_config['points'] * 0.7)
            
            db_manager.store_health_task(user_id, task_name, completed, value, points)
        
        # Also create tasks for the past few days to show trends
        for i in range(1, 8):  # Past 7 days
            past_date = today - timedelta(days=i)
            for task_name, task_config in Config.DAILY_TASKS.items():
                # Simulate some historical data
                completed = (i + hash(task_name)) % 3 != 0  # Random but deterministic
                value = task_config['target'] if completed else task_config['target'] * (0.5 + (i % 3) * 0.2)
                points = task_config['points'] if completed else int(task_config['points'] * 0.6)
                
                # Manually insert with specific date
                conn = db_manager.get_connection()
                try:
                    conn.execute('''
                        INSERT INTO health_tasks 
                        (user_id, task_type, completed, value, points_earned, date)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (user_id, task_name, completed, value, points, past_date))
                    conn.commit()
                except Exception as e:
                    logger.error(f"Error creating historical task {task_name} for {past_date}: {str(e)}")
                    conn.rollback()
                finally:
                    conn.close()
        
        logger.info("Sample health tasks created successfully")
        
        # Initialize user streaks
        db_manager.update_user_streak(user_id, 'daily_tasks', True)
        logger.info("Sample user streak initialized")
        
        logger.info("✅ Sample data created successfully!")
        
    except Exception as e:
        logger.error(f"Error creating sample data: {str(e)}")

def verify_database():
    """Verify database setup and tables"""
    logger.info("Verifying database setup...")
    
    try:
        db_manager = DatabaseManager(Config.DATABASE_URL)
        conn = db_manager.get_connection()
        
        # Check tables exist
        cursor = conn.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name NOT LIKE 'sqlite_%'
        """)
        
        tables = [row[0] for row in cursor.fetchall()]
        expected_tables = [
            'users', 'predictions', 'health_tasks', 
            'user_streaks', 'health_metrics', 'population_stats'
        ]
        
        logger.info(f"Found tables: {tables}")
        
        missing_tables = [table for table in expected_tables if table not in tables]
        if missing_tables:
            logger.error(f"Missing tables: {missing_tables}")
            return False
        
        # Check sample data
        cursor = conn.execute("SELECT COUNT(*) FROM users")
        user_count = cursor.fetchone()[0]
        
        cursor = conn.execute("SELECT COUNT(*) FROM predictions")
        prediction_count = cursor.fetchone()[0]
        
        cursor = conn.execute("SELECT COUNT(*) FROM health_tasks")
        task_count = cursor.fetchone()[0]
        
        logger.info(f"Database verification:")
        logger.info(f"  Users: {user_count}")
        logger.info(f"  Predictions: {prediction_count}")
        logger.info(f"  Health tasks: {task_count}")
        
        conn.close()
        logger.info("✅ Database verification completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Database verification failed: {str(e)}")
        return False

def reset_database():
    """Reset database (delete and recreate)"""
    logger.info("Resetting database...")
    
    try:
        # Remove existing database file
        db_path = Config.DATABASE_URL.replace('sqlite:///', '')
        if os.path.exists(db_path):
            os.remove(db_path)
            logger.info(f"Removed existing database: {db_path}")
        
        # Recreate database
        return setup_database()
        
    except Exception as e:
        logger.error(f"Database reset failed: {str(e)}")
        return False

def main():
    """Main setup script"""
    logger.info("="*60)
    logger.info("Enhanced Diabetes Predictor - Database Setup")
    logger.info("="*60)
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Setup database for Enhanced Diabetes Predictor')
    parser.add_argument('--reset', action='store_true', 
                       help='Reset database (delete and recreate)')
    parser.add_argument('--verify', action='store_true',
                       help='Verify database setup only')
    
    args = parser.parse_args()
    
    try:
        if args.verify:
            success = verify_database()
        elif args.reset:
            success = reset_database()
        else:
            success = setup_database()
        
        if success:
            logger.info("✅ Database setup completed successfully!")
            
            # Run verification
            if not args.verify:
                verify_database()
            
            logger.info("\n" + "="*60)
            logger.info("Database is ready! You can now start the application with:")
            logger.info("python app.py")
            logger.info("="*60)
            
            return 0
        else:
            logger.error("❌ Database setup failed!")
            return 1
            
    except KeyboardInterrupt:
        logger.info("Setup interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return 1

if __name__ == "__main__":
    exit(main())