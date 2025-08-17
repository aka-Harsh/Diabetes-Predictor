from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Any
import logging
import json
from config import Config

logger = logging.getLogger(__name__)

class GamificationService:
    def __init__(self):
        self.daily_tasks = Config.DAILY_TASKS
        self.achievements = self._initialize_achievements()
    
    def _initialize_achievements(self) -> Dict:
        """Initialize achievement definitions"""
        return {
            'first_prediction': {
                'title': 'Health Explorer',
                'description': 'Complete your first diabetes risk assessment',
                'points': 50,
                'icon': '🔍',
                'category': 'milestone'
            },
            'streak_7': {
                'title': 'Week Warrior',
                'description': 'Maintain health tasks for 7 consecutive days',
                'points': 100,
                'icon': '🔥',
                'category': 'streak'
            },
            'streak_30': {
                'title': 'Monthly Master',
                'description': 'Maintain health tasks for 30 consecutive days',
                'points': 500,
                'icon': '🏆',
                'category': 'streak'
            },
            'perfect_week': {
                'title': 'Perfect Week',
                'description': 'Complete all daily tasks for 7 days',
                'points': 200,
                'icon': '⭐',
                'category': 'completion'
            },
            'hydration_hero': {
                'title': 'Hydration Hero',
                'description': 'Meet water intake goals for 14 days',
                'points': 150,
                'icon': '💧',
                'category': 'specific'
            },
            'fitness_fanatic': {
                'title': 'Fitness Fanatic',
                'description': 'Complete physical activity goals for 21 days',
                'points': 300,
                'icon': '🏃',
                'category': 'specific'
            },
            'points_1000': {
                'title': 'Point Collector',
                'description': 'Earn 1000 total health points',
                'points': 100,
                'icon': '💎',
                'category': 'points'
            },
            'risk_reducer': {
                'title': 'Risk Reducer',
                'description': 'Show improvement in diabetes risk assessment',
                'points': 250,
                'icon': '📈',
                'category': 'health'
            }
        }
    
    def get_daily_tasks(self, user_id: str, target_date: date) -> Dict:
        """Get daily tasks for user with completion status"""
        try:
            from utils.database import DatabaseManager
            db = DatabaseManager(Config.DATABASE_URL)
            completed_tasks = db.get_daily_tasks(user_id, target_date)
            
            # Create task dictionary with completion status
            tasks = {}
            for task_name, task_config in self.daily_tasks.items():
                # Find if task was completed
                completed_task = next(
                    (t for t in completed_tasks if t['task_type'] == task_name), 
                    None
                )
                
                tasks[task_name] = {
                    'name': task_name,
                    'display_name': self._get_task_display_name(task_name),
                    'target': task_config['target'],
                    'unit': task_config['unit'],
                    'points': task_config['points'],
                    'completed': completed_task['completed'] if completed_task else False,
                    'current_value': completed_task['value'] if completed_task else 0,
                    'points_earned': completed_task['points_earned'] if completed_task else 0,
                    'icon': self._get_task_icon(task_name),
                    'category': self._get_task_category(task_name)
                }
            
            return {
                'date': target_date.isoformat(),
                'tasks': tasks,
                'total_possible_points': sum(config['points'] for config in self.daily_tasks.values()),
                'earned_points': sum(task['points_earned'] for task in tasks.values()),
                'completion_rate': sum(1 for task in tasks.values() if task['completed']) / len(tasks) if tasks else 0
            }
            
        except Exception as e:
            logger.error(f"Error getting daily tasks: {str(e)}")
            # Return default tasks structure instead of empty
            tasks = {}
            for task_name, task_config in self.daily_tasks.items():
                tasks[task_name] = {
                    'name': task_name,
                    'display_name': self._get_task_display_name(task_name),
                    'target': task_config['target'],
                    'unit': task_config['unit'],
                    'points': task_config['points'],
                    'completed': False,
                    'current_value': 0,
                    'points_earned': 0,
                    'icon': self._get_task_icon(task_name),
                    'category': self._get_task_category(task_name)
                }
            
            return {
                'date': datetime.now().date().isoformat(),
                'tasks': tasks,
                'total_possible_points': sum(config['points'] for config in self.daily_tasks.values()),
                'earned_points': 0,
                'completion_rate': 0
            }
    
    def update_task_completion(self, user_id: str, task_type: str, completed: bool, value: float = 0) -> Dict:
        """Update task completion and calculate points"""
        try:
            from utils.database import DatabaseManager
            
            if task_type not in self.daily_tasks:
                return {'error': f'Invalid task type: {task_type}', 'success': False}
            
            task_config = self.daily_tasks[task_type]
            
            # Calculate points based on completion and value
            if completed:
                if value >= task_config['target']:
                    points = task_config['points']
                else:
                    # Partial points for partial completion
                    points = int(task_config['points'] * (value / task_config['target']))
            else:
                points = 0
            
            # Store in database
            db = DatabaseManager(Config.DATABASE_URL)
            db.store_health_task(user_id, task_type, completed, value, points)
            
            # Update streaks
            if completed:
                self._update_task_streaks(user_id, task_type, db)
            
            # Check for new achievements
            new_achievements = self._check_achievements(user_id, db)
            
            return {
                'task_type': task_type,
                'completed': completed,
                'value': value,
                'points_earned': points,
                'new_achievements': new_achievements,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Error updating task completion: {str(e)}")
            return {'error': str(e), 'success': False}
    
    def get_current_streak(self, user_id: str) -> int:
        """Get current health streak for user"""
        try:
            from utils.database import DatabaseManager
            db = DatabaseManager(Config.DATABASE_URL)
            
            # Check consecutive days with at least one completed task
            current_date = datetime.now().date()
            streak = 0
            
            for i in range(365):  # Check up to a year back
                check_date = current_date - timedelta(days=i)
                daily_tasks = db.get_daily_tasks(user_id, check_date)
                
                # Check if any task was completed on this day
                has_completion = any(task['completed'] for task in daily_tasks)
                
                if has_completion:
                    streak += 1
                else:
                    break
            
            return streak
            
        except Exception as e:
            logger.error(f"Error getting current streak: {str(e)}")
            return 0
    
    def _update_task_streaks(self, user_id: str, task_type: str, db_manager):
        """Update streaks for specific task completion"""
        try:
            # Update general health streak
            db_manager.update_user_streak(user_id, 'daily_tasks', True)
            
            # Update task-specific streak
            db_manager.update_user_streak(user_id, f'{task_type}_streak', True)
            
        except Exception as e:
            logger.error(f"Error updating task streaks: {str(e)}")
    
    def _check_achievements(self, user_id: str, db_manager) -> List[Dict]:
        """Check for new achievements"""
        try:
            new_achievements = []
            
            # Get user stats
            stats = self.get_user_stats(user_id)
            
            # Check streak achievements
            current_streak = stats.get('current_streak', 0)
            if current_streak == 7 and not stats.get('achievements', {}).get('streak_7'):
                new_achievements.append(self.achievements['streak_7'])
            elif current_streak == 30 and not stats.get('achievements', {}).get('streak_30'):
                new_achievements.append(self.achievements['streak_30'])
            
            # Check completion achievements
            weekly_completion = stats.get('weekly_completion_rate', 0)
            if weekly_completion >= 1.0 and not stats.get('achievements', {}).get('perfect_week'):
                new_achievements.append(self.achievements['perfect_week'])
            
            # Check points achievements
            total_points = stats.get('total_points', 0)
            if total_points >= 1000 and not stats.get('achievements', {}).get('points_1000'):
                new_achievements.append(self.achievements['points_1000'])
            
            # Store new achievements in database
            for achievement in new_achievements:
                self._store_achievement(user_id, achievement, db_manager)
            
            return new_achievements
            
        except Exception as e:
            logger.error(f"Error checking achievements: {str(e)}")
            return []
    
    def _store_achievement(self, user_id: str, achievement: Dict, db_manager):
        """Store achievement in database"""
        try:
            # This would store in an achievements table
            # For simplicity, we'll log it
            logger.info(f"User {user_id} earned achievement: {achievement['title']}")
        except Exception as e:
            logger.error(f"Error storing achievement: {str(e)}")
    
    def get_user_stats(self, user_id: str) -> Dict:
        """Get comprehensive user statistics"""
        try:
            from utils.database import DatabaseManager
            db = DatabaseManager(Config.DATABASE_URL)
            
            # Get current streak
            current_streak = self.get_current_streak(user_id)
            
            # Get recent completion rates
            today = datetime.now().date()
            weekly_tasks = []
            total_points = 0
            
            for i in range(7):
                check_date = today - timedelta(days=i)
                daily_tasks = db.get_daily_tasks(user_id, check_date)
                
                if daily_tasks:
                    completed = sum(1 for task in daily_tasks if task['completed'])
                    total_tasks = len(daily_tasks)
                    completion_rate = completed / total_tasks if total_tasks > 0 else 0
                    daily_points = sum(task['points_earned'] for task in daily_tasks)
                    
                    weekly_tasks.append({
                        'date': check_date.isoformat(),
                        'completion_rate': completion_rate,
                        'points': daily_points
                    })
                    total_points += daily_points
            
            # Calculate averages
            weekly_completion_rate = sum(task['completion_rate'] for task in weekly_tasks) / len(weekly_tasks) if weekly_tasks else 0
            daily_avg_points = total_points / 7 if total_points > 0 else 0
            
            # Get best streak (would come from database)
            best_streak = current_streak  # Simplified
            
            return {
                'current_streak': current_streak,
                'best_streak': best_streak,
                'weekly_completion_rate': weekly_completion_rate,
                'total_points': total_points,
                'daily_avg_points': daily_avg_points,
                'weekly_tasks': weekly_tasks,
                'level': self._calculate_user_level(total_points),
                'next_level_points': self._get_next_level_points(total_points)
            }
            
        except Exception as e:
            logger.error(f"Error getting user stats: {str(e)}")
            return {
                'current_streak': 0,
                'best_streak': 0,
                'weekly_completion_rate': 0,
                'total_points': 0,
                'daily_avg_points': 0,
                'weekly_tasks': [],
                'level': 1,
                'next_level_points': 100
            }
    
    def _calculate_user_level(self, total_points: int) -> int:
        """Calculate user level based on total points"""
        if total_points < 100:
            return 1
        elif total_points < 300:
            return 2
        elif total_points < 600:
            return 3
        elif total_points < 1000:
            return 4
        elif total_points < 1500:
            return 5
        else:
            return 6 + (total_points - 1500) // 500
    
    def _get_next_level_points(self, total_points: int) -> int:
        """Get points needed for next level"""
        current_level = self._calculate_user_level(total_points)
        
        level_thresholds = [0, 100, 300, 600, 1000, 1500]
        
        if current_level <= len(level_thresholds):
            if current_level == len(level_thresholds):
                # For levels beyond predefined thresholds
                return 1500 + (current_level - 5) * 500
            else:
                return level_thresholds[current_level]
        else:
            return 1500 + (current_level - 5) * 500
    
    def get_leaderboard(self, limit: int = 10) -> List[Dict]:
        """Get leaderboard of top users (simplified for demo)"""
        try:
            # This would query all users and rank them
            # For demo purposes, return placeholder data
            return [
                {
                    'rank': 1,
                    'user_id': 'user_001',
                    'points': 2450,
                    'level': 7,
                    'streak': 45
                },
                {
                    'rank': 2,
                    'user_id': 'user_002',
                    'points': 1890,
                    'level': 6,
                    'streak': 23
                }
            ]
            
        except Exception as e:
            logger.error(f"Error getting leaderboard: {str(e)}")
            return []
    
    def get_weekly_challenges(self) -> List[Dict]:
        """Get current weekly challenges"""
        # This would be dynamic based on current week
        return [
            {
                'id': 'hydration_week',
                'title': 'Hydration Hero',
                'description': 'Meet your water intake goal 5 days this week',
                'target': 5,
                'current_progress': 3,
                'reward_points': 200,
                'expires': (datetime.now() + timedelta(days=4)).isoformat()
            },
            {
                'id': 'step_challenge',
                'title': 'Step Master',
                'description': 'Walk 10,000+ steps 3 days this week',
                'target': 3,
                'current_progress': 1,
                'reward_points': 300,
                'expires': (datetime.now() + timedelta(days=4)).isoformat()
            }
        ]
    
    def _get_task_display_name(self, task_name: str) -> str:
        """Get user-friendly display name for task"""
        display_names = {
            'water_intake': 'Water Intake',
            'fruit_consumption': 'Fruit Consumption',
            'physical_activity': 'Physical Activity',
            'sleep_hours': 'Sleep Hours',
            'medication_taken': 'Medication Adherence',
            'stress_management': 'Stress Management'
        }
        return display_names.get(task_name, task_name.replace('_', ' ').title())
    
    def _get_task_icon(self, task_name: str) -> str:
        """Get icon for task type"""
        icons = {
            'water_intake': '💧',
            'fruit_consumption': '🍎',
            'physical_activity': '🏃',
            'sleep_hours': '😴',
            'medication_taken': '💊',
            'stress_management': '🧘'
        }
        return icons.get(task_name, '✅')
    
    def _get_task_category(self, task_name: str) -> str:
        """Get category for task type"""
        categories = {
            'water_intake': 'nutrition',
            'fruit_consumption': 'nutrition',
            'physical_activity': 'exercise',
            'sleep_hours': 'lifestyle',
            'medication_taken': 'medical',
            'stress_management': 'wellness'
        }
        return categories.get(task_name, 'general')