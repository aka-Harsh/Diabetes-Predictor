import sqlite3
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self, database_url: str):
        self.database_url = database_url.replace('sqlite:///', '')
        self.initialize_database()
    
    def get_connection(self):
        """Get database connection"""
        conn = sqlite3.connect(self.database_url)
        conn.row_factory = sqlite3.Row
        return conn
    
    def initialize_database(self):
        """Initialize database schema"""
        conn = self.get_connection()
        try:
            # Users table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    profile_data JSON
                )
            ''')
            
            # Predictions table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    input_data JSON NOT NULL,
                    predictions JSON NOT NULL,
                    risk_analysis JSON,
                    model_version TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            ''')
            
            # Health tasks table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS health_tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    task_type TEXT NOT NULL,
                    completed BOOLEAN DEFAULT FALSE,
                    value REAL DEFAULT 0,
                    points_earned INTEGER DEFAULT 0,
                    date DATE NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            ''')
            
            # User streaks table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS user_streaks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    streak_type TEXT NOT NULL,
                    current_streak INTEGER DEFAULT 0,
                    best_streak INTEGER DEFAULT 0,
                    last_activity DATE,
                    FOREIGN KEY (user_id) REFERENCES users (id),
                    UNIQUE(user_id, streak_type)
                )
            ''')
            
            # Health metrics table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS health_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    metric_type TEXT NOT NULL,
                    value REAL NOT NULL,
                    unit TEXT,
                    date DATE NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            ''')
            
            # Population statistics table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS population_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    age_group TEXT,
                    gender TEXT,
                    avg_risk REAL,
                    total_users INTEGER,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create indexes
            conn.execute('CREATE INDEX IF NOT EXISTS idx_predictions_user_date ON predictions(user_id, created_at)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_health_tasks_user_date ON health_tasks(user_id, date)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_health_metrics_user_date ON health_metrics(user_id, date)')
            
            conn.commit()
            logger.info("Database schema initialized successfully")
        except Exception as e:
            logger.error(f"Database initialization error: {str(e)}")
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def create_user_session(self) -> str:
        """Create a new user session"""
        user_id = str(uuid.uuid4())
        conn = self.get_connection()
        try:
            conn.execute(
                'INSERT INTO users (id, profile_data) VALUES (?, ?)',
                (user_id, json.dumps({}))
            )
            conn.commit()
            return user_id
        except Exception as e:
            logger.error(f"Error creating user session: {str(e)}")
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def store_prediction(self, user_id: str, input_data: Dict, predictions: Dict, risk_analysis: Dict = None):
        """Store prediction results"""
        conn = self.get_connection()
        try:
            conn.execute('''
                INSERT INTO predictions 
                (user_id, input_data, predictions, risk_analysis, model_version)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                user_id,
                json.dumps(input_data),
                json.dumps(predictions),
                json.dumps(risk_analysis) if risk_analysis else None,
                '1.0'
            ))
            conn.commit()
        except Exception as e:
            logger.error(f"Error storing prediction: {str(e)}")
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def get_user_prediction_count(self, user_id: str) -> int:
        """Get total prediction count for user"""
        conn = self.get_connection()
        try:
            cursor = conn.execute(
                'SELECT COUNT(*) FROM predictions WHERE user_id = ?',
                (user_id,)
            )
            return cursor.fetchone()[0]
        finally:
            conn.close()
    
    def get_recent_predictions(self, user_id: str, limit: int = 5) -> List[Dict]:
        """Get recent predictions for user"""
        conn = self.get_connection()
        try:
            cursor = conn.execute('''
                SELECT input_data, predictions, risk_analysis, created_at
                FROM predictions 
                WHERE user_id = ? 
                ORDER BY created_at DESC 
                LIMIT ?
            ''', (user_id, limit))
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    'input_data': json.loads(row['input_data']),
                    'predictions': json.loads(row['predictions']),
                    'risk_analysis': json.loads(row['risk_analysis']) if row['risk_analysis'] else None,
                    'created_at': row['created_at']
                })
            return results
        finally:
            conn.close()
    
    def get_latest_prediction(self, user_id: str) -> Optional[Dict]:
        """Get latest prediction for user"""
        recent = self.get_recent_predictions(user_id, 1)
        return recent[0] if recent else None
    
    def store_health_task(self, user_id: str, task_type: str, completed: bool, value: float = 0, points: int = 0):
        """Store health task completion"""
        conn = self.get_connection()
        today = datetime.now().date()
        try:
            # Check if task already exists for today
            cursor = conn.execute('''
                SELECT id FROM health_tasks 
                WHERE user_id = ? AND task_type = ? AND date = ?
            ''', (user_id, task_type, today))
            
            existing = cursor.fetchone()
            
            if existing:
                # Update existing task
                conn.execute('''
                    UPDATE health_tasks 
                    SET completed = ?, value = ?, points_earned = ?
                    WHERE id = ?
                ''', (completed, value, points, existing['id']))
            else:
                # Insert new task
                conn.execute('''
                    INSERT INTO health_tasks 
                    (user_id, task_type, completed, value, points_earned, date)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (user_id, task_type, completed, value, points, today))
            
            conn.commit()
        except Exception as e:
            logger.error(f"Error storing health task: {str(e)}")
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def get_daily_tasks(self, user_id: str, date: datetime.date) -> List[Dict]:
        """Get daily tasks for user"""
        conn = self.get_connection()
        try:
            cursor = conn.execute('''
                SELECT task_type, completed, value, points_earned
                FROM health_tasks 
                WHERE user_id = ? AND date = ?
            ''', (user_id, date))
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    'task_type': row['task_type'],
                    'completed': bool(row['completed']),
                    'value': row['value'],
                    'points_earned': row['points_earned']
                })
            return results
        finally:
            conn.close()
    
    def update_user_streak(self, user_id: str, streak_type: str, increment: bool = True):
        """Update user streak"""
        conn = self.get_connection()
        today = datetime.now().date()
        try:
            cursor = conn.execute('''
                SELECT current_streak, best_streak, last_activity
                FROM user_streaks 
                WHERE user_id = ? AND streak_type = ?
            ''', (user_id, streak_type))
            
            existing = cursor.fetchone()
            
            if existing:
                current = existing['current_streak']
                best = existing['best_streak']
                last_activity = datetime.strptime(existing['last_activity'], '%Y-%m-%d').date() if existing['last_activity'] else None
                
                if increment:
                    if last_activity == today - timedelta(days=1):
                        current += 1
                    elif last_activity != today:
                        current = 1
                    
                    best = max(best, current)
                else:
                    current = 0
                
                conn.execute('''
                    UPDATE user_streaks 
                    SET current_streak = ?, best_streak = ?, last_activity = ?
                    WHERE user_id = ? AND streak_type = ?
                ''', (current, best, today, user_id, streak_type))
            else:
                conn.execute('''
                    INSERT INTO user_streaks 
                    (user_id, streak_type, current_streak, best_streak, last_activity)
                    VALUES (?, ?, ?, ?, ?)
                ''', (user_id, streak_type, 1 if increment else 0, 1 if increment else 0, today))
            
            conn.commit()
        except Exception as e:
            logger.error(f"Error updating streak: {str(e)}")
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def get_user_health_context(self, user_id: str) -> Dict:
        """Get user health context for AI"""
        conn = self.get_connection()
        try:
            # Get latest prediction
            latest_prediction = self.get_latest_prediction(user_id)
            
            # Get recent health tasks
            today = datetime.now().date()
            recent_tasks = self.get_daily_tasks(user_id, today)
            
            # Get current streaks
            cursor = conn.execute('''
                SELECT streak_type, current_streak, best_streak
                FROM user_streaks 
                WHERE user_id = ?
            ''', (user_id,))
            
            streaks = {}
            for row in cursor.fetchall():
                streaks[row['streak_type']] = {
                    'current': row['current_streak'],
                    'best': row['best_streak']
                }
            
            return {
                'latest_prediction': latest_prediction,
                'recent_tasks': recent_tasks,
                'streaks': streaks,
                'user_id': user_id
            }
        finally:
            conn.close()
    
    def get_health_trends(self, user_id: str, days: int = 30) -> Dict:
        """Get health trends for charts"""
        conn = self.get_connection()
        start_date = datetime.now().date() - timedelta(days=days)
        
        try:
            # Get daily task completion rates
            cursor = conn.execute('''
                SELECT date, 
                       COUNT(*) as total_tasks,
                       SUM(CASE WHEN completed THEN 1 ELSE 0 END) as completed_tasks,
                       SUM(points_earned) as daily_points
                FROM health_tasks 
                WHERE user_id = ? AND date >= ?
                GROUP BY date
                ORDER BY date
            ''', (user_id, start_date))
            
            daily_data = []
            for row in cursor.fetchall():
                completion_rate = row['completed_tasks'] / row['total_tasks'] if row['total_tasks'] > 0 else 0
                daily_data.append({
                    'date': row['date'],
                    'completion_rate': completion_rate,
                    'points': row['daily_points']
                })
            
            # Get prediction trend if multiple predictions exist
            cursor = conn.execute('''
                SELECT DATE(created_at) as date, predictions
                FROM predictions 
                WHERE user_id = ? AND created_at >= ?
                ORDER BY created_at
            ''', (user_id, start_date))
            
            prediction_trend = []
            for row in cursor.fetchall():
                try:
                    predictions = json.loads(row['predictions'])
                    # Get average risk from all models
                    risks = [pred.get('probability', 0) for pred in predictions.values() if isinstance(pred, dict)]
                    avg_risk = sum(risks) / len(risks) if risks else 0
                    prediction_trend.append({
                        'date': row['date'],
                        'risk_score': avg_risk
                    })
                except (json.JSONDecodeError, KeyError, TypeError):
                    continue
            
            return {
                'daily_completion': daily_data,
                'risk_trend': prediction_trend
            }
        except Exception as e:
            logger.error(f"Error getting health trends: {str(e)}")
            return {
                'daily_completion': [],
                'risk_trend': []
            }
        finally:
            conn.close()
    
    def get_population_statistics(self) -> Dict:
        """Get population-level statistics"""
        conn = self.get_connection()
        try:
            # Update population stats first
            self._update_population_stats(conn)
            
            cursor = conn.execute('''
                SELECT age_group, gender, avg_risk, total_users
                FROM population_stats
                ORDER BY age_group, gender
            ''')
            
            stats = {
                'age_groups': {},
                'gender_distribution': {},
                'overall_avg_risk': 0
            }
            
            total_users = 0
            total_risk = 0
            
            for row in cursor.fetchall():
                age_group = row['age_group']
                gender = row['gender']
                avg_risk = row['avg_risk']
                users = row['total_users']
                
                if age_group not in stats['age_groups']:
                    stats['age_groups'][age_group] = {}
                
                stats['age_groups'][age_group][gender] = {
                    'avg_risk': avg_risk,
                    'users': users
                }
                
                if gender not in stats['gender_distribution']:
                    stats['gender_distribution'][gender] = {'users': 0, 'avg_risk': 0}
                
                stats['gender_distribution'][gender]['users'] += users
                stats['gender_distribution'][gender]['avg_risk'] += avg_risk * users
                
                total_users += users
                total_risk += avg_risk * users
            
            # Calculate weighted averages
            for gender in stats['gender_distribution']:
                if stats['gender_distribution'][gender]['users'] > 0:
                    stats['gender_distribution'][gender]['avg_risk'] /= stats['gender_distribution'][gender]['users']
            
            stats['overall_avg_risk'] = total_risk / total_users if total_users > 0 else 0
            stats['total_users'] = total_users
            
            return stats
        finally:
            conn.close()
    
    def _update_population_stats(self, conn):
        """Update population statistics"""
        try:
            # Clear old stats
            conn.execute('DELETE FROM population_stats')
            
            # Calculate new stats from predictions
            cursor = conn.execute('''
                SELECT 
                    CASE 
                        WHEN JSON_EXTRACT(input_data, '$.age') < 30 THEN '18-29'
                        WHEN JSON_EXTRACT(input_data, '$.age') < 40 THEN '30-39'
                        WHEN JSON_EXTRACT(input_data, '$.age') < 50 THEN '40-49'
                        WHEN JSON_EXTRACT(input_data, '$.age') < 60 THEN '50-59'
                        ELSE '60+'
                    END as age_group,
                    JSON_EXTRACT(input_data, '$.gender') as gender,
                    AVG(
                        (JSON_EXTRACT(predictions, '$.random_forest.probability') + 
                         JSON_EXTRACT(predictions, '$.xgboost.probability') +
                         JSON_EXTRACT(predictions, '$.logistic_regression.probability')) / 3
                    ) as avg_risk,
                    COUNT(*) as total_users
                FROM predictions 
                WHERE created_at >= datetime('now', '-30 days')
                GROUP BY age_group, gender
            ''')
            
            for row in cursor.fetchall():
                conn.execute('''
                    INSERT INTO population_stats (age_group, gender, avg_risk, total_users)
                    VALUES (?, ?, ?, ?)
                ''', (row[0], row[1], row[2], row[3]))
            
            conn.commit()
        except Exception as e:
            logger.error(f"Error updating population stats: {str(e)}")
    
    def export_user_data(self, user_id: str) -> Dict:
        """Export all user data"""
        conn = self.get_connection()
        try:
            data = {
                'user_id': user_id,
                'export_date': datetime.now().isoformat(),
                'predictions': [],
                'health_tasks': [],
                'streaks': {},
                'health_metrics': []
            }
            
            # Get predictions
            cursor = conn.execute('''
                SELECT input_data, predictions, risk_analysis, created_at
                FROM predictions 
                WHERE user_id = ?
                ORDER BY created_at
            ''', (user_id,))
            
            for row in cursor.fetchall():
                data['predictions'].append({
                    'input_data': json.loads(row['input_data']),
                    'predictions': json.loads(row['predictions']),
                    'risk_analysis': json.loads(row['risk_analysis']) if row['risk_analysis'] else None,
                    'created_at': row['created_at']
                })
            
            # Get health tasks
            cursor = conn.execute('''
                SELECT task_type, completed, value, points_earned, date
                FROM health_tasks 
                WHERE user_id = ?
                ORDER BY date DESC
            ''', (user_id,))
            
            for row in cursor.fetchall():
                data['health_tasks'].append({
                    'task_type': row['task_type'],
                    'completed': bool(row['completed']),
                    'value': row['value'],
                    'points_earned': row['points_earned'],
                    'date': row['date']
                })
            
            # Get streaks
            cursor = conn.execute('''
                SELECT streak_type, current_streak, best_streak, last_activity
                FROM user_streaks 
                WHERE user_id = ?
            ''', (user_id,))
            
            for row in cursor.fetchall():
                data['streaks'][row['streak_type']] = {
                    'current_streak': row['current_streak'],
                    'best_streak': row['best_streak'],
                    'last_activity': row['last_activity']
                }
            
            return data
        finally:
            conn.close()