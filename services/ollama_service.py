import requests
import json
from typing import Dict, List, Optional, Any
import logging
from config import Config

logger = logging.getLogger(__name__)

class OllamaService:
    def __init__(self):
        self.base_url = Config.OLLAMA_BASE_URL
        self.model = Config.OLLAMA_MODEL
        self.health_context = self._initialize_health_context()
    
    def _initialize_health_context(self) -> str:
        """Initialize health-specific context for AI responses"""
        return """You are a helpful health assistant specializing in diabetes prevention and management. 
        You provide evidence-based, supportive guidance while always emphasizing that users should 
        consult healthcare professionals for medical decisions. Keep responses clear, encouraging, 
        and actionable. Focus on lifestyle modifications, risk factor education, and general wellness advice."""
    
    def get_health_guidance(self, user_message: str, context: Dict = None) -> str:
        """Get health guidance response from Ollama"""
        try:
            # First try to get a response from Ollama
            prompt = self._build_health_prompt(user_message, context)
            response = self._make_ollama_request(prompt)
            
            if response and 'response' in response:
                return self._format_health_response(response['response'])
            else:
                # Fallback to predefined responses for common questions
                fallback_response = self._get_fallback_response(user_message)
                if fallback_response:
                    return fallback_response
                else:
                    return "I'm having trouble connecting to the AI system right now, but I can still help! For immediate health questions, please consult with a healthcare professional. You can also try asking simpler, specific questions about diabetes prevention."
                
        except Exception as e:
            logger.error(f"Error getting health guidance: {str(e)}")
            # Try fallback response
            fallback_response = self._get_fallback_response(user_message)
            if fallback_response:
                return fallback_response
            else:
                return "I'm experiencing technical difficulties. For health questions, please consult with a healthcare professional."
    
    def _get_fallback_response(self, user_message: str) -> Optional[str]:
        """Get fallback response for common health questions when AI is unavailable"""
        message_lower = user_message.lower()
        
        # Common diabetes and health questions
        if any(word in message_lower for word in ['glucose', 'blood sugar', 'sugar level']):
            return """Normal blood glucose levels:
            
**Fasting (no food for 8+ hours):**
• Normal: Less than 100 mg/dL
• Pre-diabetes: 100-125 mg/dL  
• Diabetes: 126 mg/dL or higher

**After meals (2 hours):**
• Normal: Less than 140 mg/dL
• Pre-diabetes: 140-199 mg/dL
• Diabetes: 200 mg/dL or higher

**Random blood sugar:**
• Diabetes concern: 200 mg/dL or higher with symptoms

⚠️ Always consult your healthcare provider for proper interpretation of your specific results and treatment recommendations."""

        elif any(word in message_lower for word in ['hba1c', 'a1c', 'hemoglobin']):
            return """HbA1c (Hemoglobin A1c) levels show average blood sugar over 2-3 months:

**Normal:** Less than 5.7%
**Pre-diabetes:** 5.7% to 6.4%
**Diabetes:** 6.5% or higher

**For people with diabetes:**
• Target is usually below 7%
• Some may need stricter control (below 6.5%)
• Others may have higher targets based on individual factors

The HbA1c test is important because it shows long-term blood sugar control, not just a single moment.

⚠️ Always discuss your target HbA1c with your healthcare provider."""

        elif any(word in message_lower for word in ['diet', 'food', 'eat', 'nutrition']):
            return """Diabetes-friendly nutrition tips:

**🥗 Fill half your plate with:**
• Non-starchy vegetables (leafy greens, broccoli, peppers)
• Low glycemic fruits (berries, apples)

**🍞 Choose complex carbs:**
• Whole grains over refined grains
• Brown rice, quinoa, oats
• Monitor portion sizes

**🥩 Include lean proteins:**
• Fish, poultry, beans, tofu
• Helps stabilize blood sugar

**🥑 Healthy fats in moderation:**
• Nuts, seeds, olive oil, avocado

**💧 Stay hydrated:** Water is best, limit sugary drinks

**⏰ Timing matters:** Regular meal times help blood sugar stability

⚠️ Consider working with a registered dietitian for personalized meal planning."""

        elif any(word in message_lower for word in ['exercise', 'physical activity', 'workout']):
            return """Exercise for diabetes prevention and management:

**🎯 Weekly Goals:**
• 150 minutes moderate aerobic activity (like brisk walking)
• 2+ days of strength training

**💡 Benefits:**
• Improves insulin sensitivity
• Helps control blood sugar
• Supports weight management
• Reduces cardiovascular risk

**🚶 Good activities:**
• Walking, swimming, cycling
• Resistance training with weights or bands
• Yoga, tai chi for flexibility and stress relief

**⚠️ Safety tips:**
• Check blood sugar before/after exercise if diabetic
• Stay hydrated
• Start slowly if new to exercise
• Wear proper footwear

Always consult your healthcare provider before starting a new exercise program."""

        elif any(word in message_lower for word in ['symptoms', 'signs', 'warning']):
            return """Common diabetes symptoms to watch for:

**🚨 Classic symptoms:**
• Excessive thirst and frequent urination
• Unexplained weight loss
• Extreme fatigue
• Blurred vision
• Slow-healing cuts/infections

**⚠️ Serious symptoms (seek immediate care):**
• Severe dehydration
• Difficulty breathing
• Persistent vomiting
• Confusion or unconsciousness

**📋 Risk factors:**
• Family history of diabetes
• Overweight/obesity
• Age 45+ 
• High blood pressure
• Previous gestational diabetes

**Early detection is key!** Regular check-ups and screening can catch diabetes early when it's most manageable.

🏥 If you have concerning symptoms, contact your healthcare provider immediately."""

        elif any(word in message_lower for word in ['prevention', 'prevent', 'avoid']):
            return """Diabetes prevention strategies:

**🎯 Lifestyle changes that help:**
• Maintain healthy weight (lose 5-10% if overweight)
• Exercise regularly (150 min/week moderate activity)
• Eat a balanced, nutrient-rich diet
• Limit processed foods and added sugars
• Don't smoke
• Limit alcohol consumption
• Get adequate sleep (7-9 hours)
• Manage stress effectively

**📊 Monitor your numbers:**
• Regular blood pressure checks
• Annual glucose/HbA1c testing
• Cholesterol monitoring
• BMI tracking

**👥 High-risk individuals:**
Consider diabetes prevention programs if you have pre-diabetes or multiple risk factors.

**🔬 The good news:** Type 2 diabetes is largely preventable through lifestyle modifications!

💪 Small changes can make a big difference over time."""

        elif any(word in message_lower for word in ['medication', 'medicine', 'insulin']):
            return """About diabetes medications:

**💊 Common types:**
• Metformin (usually first-line for Type 2)
• Insulin (essential for Type 1, sometimes Type 2)
• SGLT2 inhibitors, GLP-1 agonists, etc.

**⏰ Important reminders:**
• Take medications as prescribed
• Don't skip doses
• Monitor blood sugar as directed
• Be aware of side effects

**🍎 Lifestyle still matters:**
• Medication works best with healthy diet
• Regular exercise enhances effectiveness
• Weight management remains important

**⚠️ Never stop or change medications without consulting your healthcare provider.**

**💡 Questions to ask your doctor:**
• How does this medication work?
• What side effects should I watch for?
• When should I take it?
• How will we monitor its effectiveness?

Your medication plan should be personalized to your specific needs."""

        return None  # No fallback found
    
    def _build_health_prompt(self, user_message: str, context: Dict = None) -> str:
        """Build a comprehensive prompt with health context"""
        prompt_parts = [self.health_context]
        
        # Add user context if available
        if context:
            if context.get('latest_prediction'):
                prediction = context['latest_prediction']
                risk_level = prediction.get('risk_analysis', {}).get('overall_risk_level', 'unknown')
                prompt_parts.append(f"User's current diabetes risk level: {risk_level}")
            
            if context.get('recent_tasks'):
                completed_tasks = [task['task_type'] for task in context['recent_tasks'] if task['completed']]
                if completed_tasks:
                    prompt_parts.append(f"Recently completed health tasks: {', '.join(completed_tasks)}")
            
            if context.get('streaks'):
                active_streaks = [f"{k}: {v['current']} days" for k, v in context['streaks'].items() if v['current'] > 0]
                if active_streaks:
                    prompt_parts.append(f"Current health streaks: {', '.join(active_streaks)}")
        
        # Add the user's question
        prompt_parts.append(f"User question: {user_message}")
        prompt_parts.append("Please provide a helpful, encouraging response focused on health and wellness:")
        
        return "\n\n".join(prompt_parts)
    
    def _make_ollama_request(self, prompt: str) -> Optional[Dict]:
        """Make request to Ollama API"""
        try:
            url = f"{self.base_url}/api/generate"
            
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "num_ctx": 2048,  # Reduced context for faster responses
                    "num_predict": 300  # Limit response length
                }
            }
            
            response = requests.post(
                url, 
                json=payload, 
                timeout=60,  # Increased timeout
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                return None
                
        except requests.exceptions.Timeout as e:
            logger.error(f"Ollama request timeout: {str(e)}")
            return None
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Ollama connection error: {str(e)}")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama request error: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in Ollama request: {str(e)}")
            return None
    
    def _format_health_response(self, response: str) -> str:
        """Format and validate health response"""
        # Clean up response
        response = response.strip()
        
        # Add medical disclaimer if not present
        disclaimer_keywords = ['doctor', 'physician', 'healthcare professional', 'medical professional']
        has_disclaimer = any(keyword in response.lower() for keyword in disclaimer_keywords)
        
        if not has_disclaimer and len(response) > 100:
            response += "\n\n⚠️ Remember to consult with a healthcare professional for personalized medical advice."
        
        return response
    
    def explain_prediction_results(self, predictions: Dict, risk_factors: Dict) -> str:
        """Explain prediction results in user-friendly terms"""
        try:
            # Get ensemble or best prediction
            if 'ensemble' in predictions:
                main_prediction = predictions['ensemble']
            else:
                main_prediction = list(predictions.values())[0]
            
            risk_level = main_prediction.get('risk_level', 'unknown')
            probability = main_prediction.get('probability', 0)
            
            # Build explanation prompt
            prompt = f"""
            {self.health_context}
            
            Please explain diabetes risk assessment results to a user in clear, encouraging terms:
            
            Risk Level: {risk_level}
            Risk Probability: {probability:.1%}
            
            Key factors contributing to risk:
            {self._format_risk_factors(risk_factors)}
            
            Provide an encouraging explanation that:
            1. Explains what this risk level means
            2. Emphasizes that risk can be modified
            3. Suggests practical next steps
            4. Maintains a positive, supportive tone
            """
            
            response = self._make_ollama_request(prompt)
            
            if response and 'response' in response:
                return self._format_health_response(response['response'])
            else:
                return self._get_default_explanation(risk_level, probability)
                
        except Exception as e:
            logger.error(f"Error explaining prediction: {str(e)}")
            return self._get_default_explanation('moderate', 0.5)
    
    def _format_risk_factors(self, risk_factors: Dict) -> str:
        """Format risk factors for AI prompt"""
        if not risk_factors or 'individual_factors' not in risk_factors:
            return "No specific risk factor analysis available."
        
        factors = []
        for factor, analysis in risk_factors['individual_factors'].items():
            if analysis['risk_score'] > 0.3:  # Only include significant factors
                factors.append(f"- {factor}: {analysis['severity']} risk")
        
        return '\n'.join(factors) if factors else "Most risk factors are in healthy ranges."
    
    def _get_default_explanation(self, risk_level: str, probability: float) -> str:
        """Get default explanation when AI is unavailable"""
        explanations = {
            'low': f"Great news! Your diabetes risk is currently low ({probability:.1%}). This means you're doing many things right for your health. Continue maintaining healthy habits like regular exercise, balanced nutrition, and routine health checkups.",
            
            'moderate': f"Your diabetes risk is moderate ({probability:.1%}). This is manageable with focused lifestyle improvements. Consider working with a healthcare provider to develop a prevention plan focusing on diet, exercise, and regular monitoring.",
            
            'high': f"Your diabetes risk is elevated ({probability:.1%}). Don't be discouraged - this is your opportunity to take proactive steps. Lifestyle modifications can significantly reduce your risk. Please consult with a healthcare professional to create a comprehensive prevention strategy.",
            
            'very_high': f"Your diabetes risk is quite high ({probability:.1%}), but remember that risk can be modified through dedicated lifestyle changes. It's important to work closely with a healthcare team to develop an intensive prevention plan. Many people successfully reduce their risk with proper guidance and commitment."
        }
        
        base_explanation = explanations.get(risk_level, f"Your diabetes risk level is {risk_level} with a {probability:.1%} probability.")
        
        return base_explanation + "\n\n⚠️ Please consult with a healthcare professional for personalized medical advice and treatment planning."
    
    def get_lifestyle_advice(self, user_data: Dict, improvement_area: str) -> str:
        """Get specific lifestyle advice for improvement areas"""
        try:
            advice_prompts = {
                'diet': f"""
                {self.health_context}
                
                Provide specific, actionable dietary advice for diabetes prevention for someone with:
                - Age: {user_data.get('age', 'unknown')}
                - BMI: {user_data.get('bmi', 'unknown')}
                - Current eating habits: {user_data.get('dietary_habits', 'not specified')}
                
                Focus on practical meal planning, portion control, and food choices.
                """,
                
                'exercise': f"""
                {self.health_context}
                
                Provide safe, progressive exercise recommendations for diabetes prevention for someone with:
                - Age: {user_data.get('age', 'unknown')}
                - Current activity level: {user_data.get('activity_level', 'unknown')}
                - Any physical limitations: {user_data.get('limitations', 'none specified')}
                
                Include both cardio and strength training suggestions with safety considerations.
                """,
                
                'weight_management': f"""
                {self.health_context}
                
                Provide sustainable weight management strategies for diabetes prevention:
                - Current BMI: {user_data.get('bmi', 'unknown')}
                - Weight goals: {user_data.get('weight_goal', 'general health improvement')}
                
                Focus on gradual, sustainable approaches rather than quick fixes.
                """,
                
                'stress_management': f"""
                {self.health_context}
                
                Provide stress management techniques that can help with diabetes prevention:
                - Current stress level: {user_data.get('stress_level', 'moderate')}
                - Available time for stress management: {user_data.get('available_time', 'varies')}
                
                Include both quick daily techniques and longer-term stress reduction strategies.
                """
            }
            
            prompt = advice_prompts.get(improvement_area, f"""
            {self.health_context}
            
            Provide general lifestyle advice for diabetes prevention focusing on {improvement_area}.
            Make the advice practical and encouraging.
            """)
            
            response = self._make_ollama_request(prompt)
            
            if response and 'response' in response:
                return self._format_health_response(response['response'])
            else:
                return self._get_default_lifestyle_advice(improvement_area)
                
        except Exception as e:
            logger.error(f"Error getting lifestyle advice: {str(e)}")
            return self._get_default_lifestyle_advice(improvement_area)
    
    def _get_default_lifestyle_advice(self, improvement_area: str) -> str:
        """Get default lifestyle advice when AI is unavailable"""
        default_advice = {
            'diet': """
            For diabetes prevention through diet:
            • Choose whole grains over refined carbohydrates
            • Fill half your plate with non-starchy vegetables
            • Include lean protein sources with each meal
            • Limit added sugars and processed foods
            • Practice portion control using smaller plates
            • Stay hydrated with water instead of sugary drinks
            """,
            
            'exercise': """
            For diabetes prevention through exercise:
            • Aim for 150 minutes of moderate activity weekly
            • Include strength training exercises 2-3 times per week
            • Start slowly and gradually increase intensity
            • Try activities you enjoy to maintain consistency
            • Consider walking after meals to help blood sugar control
            • Mix cardio with resistance exercises for best results
            """,
            
            'weight_management': """
            For healthy weight management:
            • Focus on gradual weight loss (1-2 pounds per week)
            • Create a modest caloric deficit through diet and exercise
            • Track your food intake and physical activity
            • Set realistic, achievable goals
            • Build sustainable habits rather than following fad diets
            • Consider working with a registered dietitian
            """,
            
            'stress_management': """
            For effective stress management:
            • Practice deep breathing exercises daily
            • Try meditation or mindfulness techniques
            • Ensure adequate sleep (7-9 hours per night)
            • Engage in regular physical activity
            • Maintain social connections and support systems
            • Consider professional counseling if stress is overwhelming
            """
        }
        
        advice = default_advice.get(improvement_area, "Focus on balanced nutrition, regular exercise, adequate sleep, and stress management for optimal health.")
        return advice + "\n\n⚠️ Please consult with healthcare professionals for personalized guidance."
    
    def generate_motivational_message(self, user_progress: Dict) -> str:
        """Generate motivational message based on user progress"""
        try:
            current_streak = user_progress.get('current_streak', 0)
            completion_rate = user_progress.get('completion_rate', 0)
            recent_improvements = user_progress.get('improvements', [])
            
            prompt = f"""
            {self.health_context}
            
            Generate an encouraging, personalized motivational message for a user with:
            - Current health task streak: {current_streak} days
            - Task completion rate: {completion_rate:.1%}
            - Recent improvements: {', '.join(recent_improvements) if recent_improvements else 'maintaining current habits'}
            
            Make the message:
            1. Specific to their progress
            2. Encouraging and positive
            3. Include actionable next steps
            4. Keep it concise but meaningful
            """
            
            response = self._make_ollama_request(prompt)
            
            if response and 'response' in response:
                return response['response'].strip()
            else:
                return self._get_default_motivational_message(current_streak, completion_rate)
                
        except Exception as e:
            logger.error(f"Error generating motivational message: {str(e)}")
            return self._get_default_motivational_message(0, 0.5)
    
    def _get_default_motivational_message(self, streak: int, completion_rate: float) -> str:
        """Get default motivational message"""
        if streak >= 30:
            return f"🏆 Incredible! {streak} days of consistent health habits. You're truly committed to your wellness journey. Keep up this amazing momentum!"
        elif streak >= 7:
            return f"🔥 Great job maintaining your health streak for {streak} days! You're building powerful habits that will benefit you long-term."
        elif completion_rate >= 0.8:
            return f"⭐ You're completing {completion_rate:.0%} of your health tasks - that's excellent! Small consistent steps lead to big health improvements."
        elif completion_rate >= 0.5:
            return f"👍 You're making good progress with {completion_rate:.0%} task completion. Every healthy choice matters - keep building on this foundation!"
        else:
            return "🌱 Every journey starts with a single step. Your health matters, and today is a great day to make positive changes. You've got this!"
    
    def check_ollama_connection(self) -> Dict:
        """Check if Ollama service is available and responsive"""
        try:
            url = f"{self.base_url}/api/tags"
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [model.get('name', '') for model in models]
                
                return {
                    'connected': True,
                    'available_models': model_names,
                    'current_model': self.model,
                    'model_available': self.model in model_names
                }
            else:
                return {
                    'connected': False,
                    'error': f'HTTP {response.status_code}',
                    'available_models': [],
                    'current_model': self.model,
                    'model_available': False
                }
                
        except Exception as e:
            return {
                'connected': False,
                'error': str(e),
                'available_models': [],
                'current_model': self.model,
                'model_available': False
            }
    
    def get_health_education_content(self, topic: str) -> str:
        """Get educational content about health topics"""
        try:
            educational_prompts = {
                'diabetes_basics': f"""
                {self.health_context}
                
                Explain diabetes basics in simple, clear terms:
                - What is diabetes?
                - Types of diabetes
                - Risk factors
                - Prevention strategies
                
                Make it educational but not overwhelming, suitable for someone just learning about diabetes.
                """,
                
                'blood_sugar_management': f"""
                {self.health_context}
                
                Explain blood sugar management for diabetes prevention:
                - What are normal blood sugar levels?
                - How does diet affect blood sugar?
                - Role of exercise in blood sugar control
                - When to monitor blood sugar
                
                Focus on practical, actionable information.
                """,
                
                'lifestyle_prevention': f"""
                {self.health_context}
                
                Explain how lifestyle changes can prevent diabetes:
                - Diet modifications that help
                - Exercise recommendations
                - Weight management strategies
                - Stress management importance
                
                Emphasize that prevention is possible and effective.
                """
            }
            
            prompt = educational_prompts.get(topic, f"""
            {self.health_context}
            
            Provide educational information about {topic} related to diabetes prevention and health.
            Make it informative, accurate, and easy to understand.
            """)
            
            response = self._make_ollama_request(prompt)
            
            if response and 'response' in response:
                return self._format_health_response(response['response'])
            else:
                return self._get_default_education_content(topic)
                
        except Exception as e:
            logger.error(f"Error getting education content: {str(e)}")
            return self._get_default_education_content(topic)
    
    def _get_default_education_content(self, topic: str) -> str:
        """Get default educational content when AI is unavailable"""
        default_content = {
            'diabetes_basics': """
            Diabetes is a condition where blood sugar levels become too high. There are two main types:
            
            Type 1: The body doesn't produce insulin (usually develops in childhood)
            Type 2: The body doesn't use insulin effectively (more common in adults)
            
            Risk factors include family history, age, weight, physical inactivity, and certain health conditions.
            
            The good news is that Type 2 diabetes can often be prevented or delayed through healthy lifestyle choices.
            """,
            
            'blood_sugar_management': """
            Normal blood sugar levels:
            • Fasting: Less than 100 mg/dL
            • After meals: Less than 140 mg/dL
            
            Diet affects blood sugar by:
            • Carbohydrates raise blood sugar most directly
            • Protein and fat have smaller effects
            • Fiber helps slow sugar absorption
            
            Exercise helps by making muscles use glucose for energy and improving insulin sensitivity.
            """,
            
            'lifestyle_prevention': """
            Lifestyle changes that help prevent diabetes:
            
            Diet: Focus on whole foods, control portions, limit processed foods
            Exercise: Aim for 150 minutes of moderate activity weekly
            Weight: Losing even 5-10% of body weight can significantly reduce risk
            Sleep: Get 7-9 hours of quality sleep nightly
            Stress: Practice stress management techniques regularly
            
            These changes work together to improve your body's ability to regulate blood sugar.
            """
        }
        
        content = default_content.get(topic, f"Educational content about {topic} is not available at this time.")
        return content + "\n\n⚠️ Always consult healthcare professionals for complete and personalized medical information."