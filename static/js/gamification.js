// Enhanced Diabetes Predictor - Gamification UI
class GamificationUI {
    constructor() {
        this.achievements = new Map();
        this.streaks = new Map();
        this.currentLevel = 1;
        this.currentPoints = 0;
        this.init();
    }

    init() {
        this.loadUserStats();
        this.setupEventListeners();
        this.initializeAnimations();
    }

    async loadUserStats() {
        try {
            const api = new APIClient();
            const stats = await api.get('/gamification/stats');
            this.updateStats(stats);
        } catch (error) {
            console.log('Gamification stats not available yet');
        }
    }

    setupEventListeners() {
        // Listen for task updates
        document.addEventListener('taskUpdated', (event) => {
            this.handleTaskUpdate(event.detail);
        });

        // Listen for achievement unlocks
        document.addEventListener('achievementUnlocked', (event) => {
            this.showAchievementNotification(event.detail);
        });

        // Level up celebrations
        document.addEventListener('levelUp', (event) => {
            this.celebrateLevelUp(event.detail);
        });
    }

    initializeAnimations() {
        // Initialize particle system for celebrations
        this.particleSystem = new ParticleSystem();
    }

    updateStats(stats) {
        this.currentLevel = stats.level || 1;
        this.currentPoints = stats.total_points || 0;
        this.currentStreak = stats.current_streak || 0;
        this.bestStreak = stats.best_streak || 0;

        this.updateUI();
    }

    updateUI() {
        // Update streak display
        this.updateStreakDisplay();
        
        // Update level progress
        this.updateLevelProgress();
        
        // Update achievements
        this.updateAchievements();
        
        // Update health score visualization
        this.updateHealthScoreDisplay();
    }

    updateStreakDisplay() {
        const streakElements = document.querySelectorAll('[data-streak]');
        streakElements.forEach(element => {
            const streakType = element.dataset.streak;
            const streakValue = this.streaks.get(streakType) || this.currentStreak;
            
            element.textContent = streakValue;
            
            // Add flame animation for active streaks
            if (streakValue > 0) {
                element.classList.add('streak-active');
                this.animateStreak(element, streakValue);
            } else {
                element.classList.remove('streak-active');
            }
        });
    }

    animateStreak(element, streakValue) {
        // Add flame emoji with animation
        const flame = element.querySelector('.streak-flame');
        if (!flame) {
            const flameSpan = document.createElement('span');
            flameSpan.className = 'streak-flame';
            flameSpan.textContent = '🔥';
            element.appendChild(flameSpan);
        }

        // Intensity based on streak length
        let intensity = 'normal';
        if (streakValue >= 30) intensity = 'legendary';
        else if (streakValue >= 14) intensity = 'epic';
        else if (streakValue >= 7) intensity = 'strong';

        element.className = `streak-counter streak-${intensity}`;
    }

    updateLevelProgress() {
        const levelElements = document.querySelectorAll('[data-level]');
        const pointsElements = document.querySelectorAll('[data-points]');
        const progressElements = document.querySelectorAll('[data-level-progress]');

        levelElements.forEach(el => el.textContent = this.currentLevel);
        pointsElements.forEach(el => el.textContent = this.currentPoints.toLocaleString());

        progressElements.forEach(element => {
            const nextLevelPoints = this.getNextLevelPoints();
            const currentLevelPoints = this.getCurrentLevelPoints();
            const progress = ((this.currentPoints - currentLevelPoints) / (nextLevelPoints - currentLevelPoints)) * 100;
            
            this.animateProgressBar(element, Math.min(progress, 100));
        });
    }

    updateAchievements() {
        const achievementsContainer = document.getElementById('achievements-container');
        if (!achievementsContainer) return;

        // Clear existing achievements
        achievementsContainer.innerHTML = '';

        // Display recent achievements
        this.achievements.forEach((achievement, id) => {
            const achievementElement = this.createAchievementElement(achievement);
            achievementsContainer.appendChild(achievementElement);
        });
    }

    updateHealthScoreDisplay() {
        const healthScoreElements = document.querySelectorAll('[data-health-score]');
        healthScoreElements.forEach(element => {
            const score = parseFloat(element.dataset.healthScore) || 0;
            this.animateHealthScore(element, score);
        });
    }

    createAchievementElement(achievement) {
        const element = document.createElement('div');
        element.className = `achievement-badge ${achievement.isNew ? 'new' : ''}`;
        
        element.innerHTML = `
            <div class="flex items-center p-3 bg-gradient-to-r from-yellow-400 to-orange-400 rounded-lg shadow-md">
                <span class="text-2xl mr-3">${achievement.icon}</span>
                <div>
                    <div class="font-semibold text-yellow-900">${achievement.title}</div>
                    <div class="text-xs text-yellow-800">${achievement.description}</div>
                    <div class="text-xs text-yellow-700 mt-1">+${achievement.points} points</div>
                </div>
            </div>
        `;

        // Remove new flag after animation
        if (achievement.isNew) {
            setTimeout(() => {
                achievement.isNew = false;
                element.classList.remove('new');
            }, 3000);
        }

        return element;
    }

    handleTaskUpdate(taskData) {
        if (taskData.completed) {
            this.celebrateTaskCompletion(taskData);
        }

        // Update streak if applicable
        if (taskData.streakUpdate) {
            this.updateStreak(taskData.taskType, taskData.streakUpdate);
        }

        // Check for level up
        if (taskData.pointsEarned) {
            this.addPoints(taskData.pointsEarned);
        }

        // Handle new achievements
        if (taskData.newAchievements) {
            taskData.newAchievements.forEach(achievement => {
                this.unlockAchievement(achievement);
            });
        }
    }

    celebrateTaskCompletion(taskData) {
        const taskElement = document.querySelector(`[data-task="${taskData.taskType}"]`);
        
        if (taskElement) {
            // Add completion animation
            taskElement.classList.add('task-completed');
            
            // Show points animation
            this.showPointsAnimation(taskElement, taskData.pointsEarned);
            
            // Confetti effect for significant milestones
            if (taskData.pointsEarned >= 20) {
                this.triggerConfetti(taskElement);
            }
        }

        // Update task completion sound
        this.playCompletionSound('task');
    }

    showPointsAnimation(element, points) {
        const pointsEl = document.createElement('div');
        pointsEl.className = 'points-animation';
        pointsEl.textContent = `+${points}`;
        pointsEl.style.cssText = `
            position: absolute;
            color: #10B981;
            font-weight: bold;
            font-size: 1.2rem;
            pointer-events: none;
            z-index: 1000;
            animation: pointsFloat 2s ease-out forwards;
        `;

        const rect = element.getBoundingClientRect();
        pointsEl.style.left = rect.right + 'px';
        pointsEl.style.top = rect.top + 'px';

        document.body.appendChild(pointsEl);

        setTimeout(() => pointsEl.remove(), 2000);
    }

    unlockAchievement(achievement) {
        achievement.isNew = true;
        this.achievements.set(achievement.id, achievement);
        
        this.showAchievementNotification(achievement);
        this.playCompletionSound('achievement');
        
        // Trigger achievement event
        document.dispatchEvent(new CustomEvent('achievementUnlocked', {
            detail: achievement
        }));
    }

    showAchievementNotification(achievement) {
        const notification = document.createElement('div');
        notification.className = 'achievement-notification fixed top-4 right-4 z-50';
        
        notification.innerHTML = `
            <div class="bg-gradient-to-r from-yellow-400 to-orange-400 text-yellow-900 p-4 rounded-lg shadow-lg transform transition-all duration-500 translate-x-full">
                <div class="flex items-center">
                    <span class="text-3xl mr-3">${achievement.icon}</span>
                    <div>
                        <div class="font-bold">Achievement Unlocked!</div>
                        <div class="font-semibold">${achievement.title}</div>
                        <div class="text-sm opacity-90">${achievement.description}</div>
                    </div>
                </div>
            </div>
        `;

        document.body.appendChild(notification);

        // Animate in
        setTimeout(() => {
            notification.firstElementChild.classList.remove('translate-x-full');
        }, 100);

        // Auto-remove
        setTimeout(() => {
            notification.firstElementChild.classList.add('translate-x-full');
            setTimeout(() => notification.remove(), 500);
        }, 5000);

        // Trigger confetti
        this.triggerConfetti(notification);
    }

    celebrateLevelUp(levelData) {
        const modal = document.createElement('div');
        modal.className = 'level-up-modal fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50';
        
        modal.innerHTML = `
            <div class="bg-white rounded-2xl p-8 text-center max-w-md mx-4 transform transition-all duration-500 scale-90 opacity-0">
                <div class="text-6xl mb-4">🎉</div>
                <h2 class="text-3xl font-bold text-gray-900 mb-2">Level Up!</h2>
                <div class="text-xl text-blue-600 mb-4">You reached Level ${levelData.newLevel}</div>
                <div class="text-gray-600 mb-6">${levelData.message || 'Keep up the great work!'}</div>
                <button onclick="this.parentElement.parentElement.remove()" 
                        class="bg-blue-600 text-white px-6 py-3 rounded-lg hover:bg-blue-700 transition-colors duration-200">
                    Awesome!
                </button>
            </div>
        `;

        document.body.appendChild(modal);

        // Animate in
        setTimeout(() => {
            const content = modal.firstElementChild;
            content.classList.remove('scale-90', 'opacity-0');
        }, 100);

        // Full screen confetti
        this.triggerFullScreenConfetti();
        this.playCompletionSound('levelup');
    }

    addPoints(points) {
        const oldLevel = this.currentLevel;
        this.currentPoints += points;
        
        // Check for level up
        const newLevel = this.calculateLevel(this.currentPoints);
        if (newLevel > oldLevel) {
            this.currentLevel = newLevel;
            this.celebrateLevelUp({
                oldLevel: oldLevel,
                newLevel: newLevel,
                message: `Congratulations! You've reached Level ${newLevel}!`
            });
        }

        this.updateLevelProgress();
    }

    updateStreak(streakType, streakData) {
        this.streaks.set(streakType, streakData.current);
        
        // Celebrate streak milestones
        if (streakData.current > 0 && streakData.current % 7 === 0) {
            this.celebrateStreakMilestone(streakType, streakData.current);
        }

        this.updateStreakDisplay();
    }

    celebrateStreakMilestone(streakType, streakValue) {
        const message = `${streakValue} day streak! You're on fire! 🔥`;
        
        if (typeof showToast === 'function') {
            showToast(message, 'success', 4000);
        }

        // Special effects for major milestones
        if (streakValue >= 30) {
            this.triggerConfetti();
            this.playCompletionSound('milestone');
        }
    }

    animateProgressBar(element, percentage) {
        element.style.width = '0%';
        element.style.transition = 'width 1s ease-in-out';
        
        setTimeout(() => {
            element.style.width = percentage + '%';
        }, 100);
    }

    animateHealthScore(element, score) {
        let currentScore = 0;
        const increment = score / 50; // 50 frames
        
        const animate = () => {
            currentScore += increment;
            if (currentScore < score) {
                element.textContent = Math.round(currentScore);
                requestAnimationFrame(animate);
            } else {
                element.textContent = Math.round(score);
            }
        };
        
        animate();
    }

    triggerConfetti(sourceElement = null) {
        if (this.particleSystem) {
            const rect = sourceElement ? sourceElement.getBoundingClientRect() : null;
            this.particleSystem.burst(rect);
        }
    }

    triggerFullScreenConfetti() {
        if (this.particleSystem) {
            this.particleSystem.celebration();
        }
    }

    playCompletionSound(type) {
        // Simple audio feedback using Web Audio API
        try {
            const audioContext = new (window.AudioContext || window.webkitAudioContext)();
            const oscillator = audioContext.createOscillator();
            const gainNode = audioContext.createGain();

            oscillator.connect(gainNode);
            gainNode.connect(audioContext.destination);

            // Different tones for different events
            const frequencies = {
                task: [523.25, 659.25], // C5, E5
                achievement: [523.25, 659.25, 783.99], // C5, E5, G5
                levelup: [523.25, 659.25, 783.99, 1046.50], // C5, E5, G5, C6
                milestone: [392.00, 493.88, 659.25] // G4, B4, E5
            };

            const notes = frequencies[type] || frequencies.task;
            
            notes.forEach((freq, index) => {
                setTimeout(() => {
                    const osc = audioContext.createOscillator();
                    const gain = audioContext.createGain();
                    
                    osc.connect(gain);
                    gain.connect(audioContext.destination);
                    
                    osc.frequency.value = freq;
                    osc.type = 'sine';
                    
                    gain.gain.setValueAtTime(0.1, audioContext.currentTime);
                    gain.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.3);
                    
                    osc.start(audioContext.currentTime);
                    osc.stop(audioContext.currentTime + 0.3);
                }, index * 100);
            });
        } catch (error) {
            // Audio not supported or blocked
            console.log('Audio feedback not available');
        }
    }

    calculateLevel(points) {
        if (points < 100) return 1;
        if (points < 300) return 2;
        if (points < 600) return 3;
        if (points < 1000) return 4;
        if (points < 1500) return 5;
        return 6 + Math.floor((points - 1500) / 500);
    }

    getCurrentLevelPoints() {
        const thresholds = [0, 100, 300, 600, 1000, 1500];
        if (this.currentLevel <= thresholds.length) {
            return thresholds[this.currentLevel - 1];
        }
        return 1500 + (this.currentLevel - 6) * 500;
    }

    getNextLevelPoints() {
        const thresholds = [0, 100, 300, 600, 1000, 1500];
        if (this.currentLevel < thresholds.length) {
            return thresholds[this.currentLevel];
        }
        return 1500 + (this.currentLevel - 5) * 500;
    }
}

// Simple particle system for celebrations
class ParticleSystem {
    constructor() {
        this.canvas = this.createCanvas();
        this.ctx = this.canvas.getContext('2d');
        this.particles = [];
        this.animationId = null;
    }

    createCanvas() {
        const canvas = document.createElement('canvas');
        canvas.style.position = 'fixed';
        canvas.style.top = '0';
        canvas.style.left = '0';
        canvas.style.width = '100%';
        canvas.style.height = '100%';
        canvas.style.pointerEvents = 'none';
        canvas.style.zIndex = '9999';
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
        document.body.appendChild(canvas);
        return canvas;
    }

    burst(sourceRect = null) {
        const colors = ['#FFD700', '#FF6347', '#32CD32', '#1E90FF', '#FF69B4'];
        const particleCount = 20;
        
        const centerX = sourceRect ? sourceRect.left + sourceRect.width / 2 : window.innerWidth / 2;
        const centerY = sourceRect ? sourceRect.top + sourceRect.height / 2 : window.innerHeight / 2;
        
        for (let i = 0; i < particleCount; i++) {
            this.particles.push({
                x: centerX,
                y: centerY,
                vx: (Math.random() - 0.5) * 10,
                vy: (Math.random() - 0.5) * 10 - 5,
                color: colors[Math.floor(Math.random() * colors.length)],
                size: Math.random() * 6 + 2,
                life: 1,
                decay: Math.random() * 0.02 + 0.01
            });
        }
        
        this.animate();
    }

    celebration() {
        const colors = ['#FFD700', '#FF6347', '#32CD32', '#1E90FF', '#FF69B4', '#FFA500'];
        const particleCount = 100;
        
        for (let i = 0; i < particleCount; i++) {
            this.particles.push({
                x: Math.random() * window.innerWidth,
                y: -10,
                vx: (Math.random() - 0.5) * 4,
                vy: Math.random() * 5 + 2,
                color: colors[Math.floor(Math.random() * colors.length)],
                size: Math.random() * 8 + 3,
                life: 1,
                decay: Math.random() * 0.01 + 0.005
            });
        }
        
        this.animate();
    }

    animate() {
        if (this.animationId) return;
        
        this.animationId = requestAnimationFrame(() => this.updateParticles());
    }

    updateParticles() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        for (let i = this.particles.length - 1; i >= 0; i--) {
            const particle = this.particles[i];
            
            particle.x += particle.vx;
            particle.y += particle.vy;
            particle.vy += 0.2; // gravity
            particle.life -= particle.decay;
            
            if (particle.life <= 0) {
                this.particles.splice(i, 1);
                continue;
            }
            
            this.ctx.save();
            this.ctx.globalAlpha = particle.life;
            this.ctx.fillStyle = particle.color;
            this.ctx.beginPath();
            this.ctx.arc(particle.x, particle.y, particle.size, 0, Math.PI * 2);
            this.ctx.fill();
            this.ctx.restore();
        }
        
        if (this.particles.length > 0) {
            this.animationId = requestAnimationFrame(() => this.updateParticles());
        } else {
            this.animationId = null;
        }
    }
}

// Global gamification functions
function updateTaskProgress(taskType, value, target) {
    const progressElement = document.querySelector(`[data-task-progress="${taskType}"]`);
    if (progressElement) {
        const percentage = Math.min((value / target) * 100, 100);
        progressElement.style.width = percentage + '%';
        
        // Add completion class if target is met
        const taskElement = progressElement.closest('[data-task]');
        if (taskElement) {
            if (percentage >= 100) {
                taskElement.classList.add('task-completed');
            } else {
                taskElement.classList.remove('task-completed');
            }
        }
    }
}

function showStreakUpdate(streakType, currentStreak, isNewRecord = false) {
    const streakElement = document.querySelector(`[data-streak="${streakType}"]`);
    if (streakElement) {
        streakElement.textContent = currentStreak;
        
        if (isNewRecord) {
            streakElement.classList.add('streak-record');
            setTimeout(() => {
                streakElement.classList.remove('streak-record');
            }, 2000);
        }
    }
}

function displayLevelProgress(currentPoints, currentLevel, nextLevelPoints) {
    const levelElements = document.querySelectorAll('[data-current-level]');
    const pointsElements = document.querySelectorAll('[data-current-points]');
    const progressElements = document.querySelectorAll('[data-level-progress]');
    
    levelElements.forEach(el => el.textContent = currentLevel);
    pointsElements.forEach(el => el.textContent = currentPoints.toLocaleString());
    
    const currentLevelStart = getLevelStartPoints(currentLevel);
    const progress = ((currentPoints - currentLevelStart) / (nextLevelPoints - currentLevelStart)) * 100;
    
    progressElements.forEach(element => {
        element.style.width = Math.min(progress, 100) + '%';
    });
}

function getLevelStartPoints(level) {
    const thresholds = [0, 100, 300, 600, 1000, 1500];
    if (level <= thresholds.length) {
        return thresholds[level - 1];
    }
    return 1500 + (level - 6) * 500;
}

// Achievement templates
const AchievementTemplates = {
    firstPrediction: {
        id: 'first_prediction',
        title: 'Health Explorer',
        description: 'Complete your first diabetes risk assessment',
        icon: '🔍',
        points: 50,
        category: 'milestone'
    },
    
    weekStreak: {
        id: 'week_streak',
        title: 'Week Warrior',
        description: 'Maintain health tasks for 7 consecutive days',
        icon: '🔥',
        points: 100,
        category: 'streak'
    },
    
    monthStreak: {
        id: 'month_streak',
        title: 'Monthly Master',
        description: 'Maintain health tasks for 30 consecutive days',
        icon: '🏆',
        points: 500,
        category: 'streak'
    },
    
    perfectWeek: {
        id: 'perfect_week',
        title: 'Perfect Week',
        description: 'Complete all daily tasks for 7 days',
        icon: '⭐',
        points: 200,
        category: 'completion'
    },
    
    hydrationHero: {
        id: 'hydration_hero',
        title: 'Hydration Hero',
        description: 'Meet water intake goals for 14 days',
        icon: '💧',
        points: 150,
        category: 'specific'
    },
    
    fitnessLegend: {
        id: 'fitness_legend',
        title: 'Fitness Legend',
        description: 'Complete physical activity goals for 21 days',
        icon: '🏃',
        points: 300,
        category: 'specific'
    }
};

// Initialize gamification when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    if (typeof GamificationUI !== 'undefined') {
        window.gamificationUI = new GamificationUI();
    }
});

// CSS animations for gamification (injected styles)
const gamificationStyles = `
<style>
@keyframes pointsFloat {
    0% {
        opacity: 1;
        transform: translateY(0px);
    }
    100% {
        opacity: 0;
        transform: translateY(-50px);
    }
}

@keyframes achievementPulse {
    0%, 100% {
        transform: scale(1);
    }
    50% {
        transform: scale(1.05);
    }
}

@keyframes streakFlicker {
    0%, 100% {
        opacity: 1;
        transform: scale(1);
    }
    50% {
        opacity: 0.8;
        transform: scale(0.95);
    }
}

.streak-active .streak-flame {
    animation: streakFlicker 2s infinite alternate;
}

.streak-normal {
    color: #F59E0B;
}

.streak-strong {
    color: #EF4444;
}

.streak-epic {
    color: #8B5CF6;
    text-shadow: 0 0 10px rgba(139, 92, 246, 0.5);
}

.streak-legendary {
    background: linear-gradient(45deg, #FFD700, #FF6347);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-shadow: 0 0 20px rgba(255, 215, 0, 0.8);
    animation: achievementPulse 1s infinite;
}

.streak-record {
    animation: achievementPulse 2s ease-in-out;
    color: #10B981;
    font-weight: bold;
}

.task-completed {
    background-color: rgba(16, 185, 129, 0.1);
    border-color: #10B981;
    transform: scale(1.02);
    transition: all 0.3s ease;
}

.task-completed .task-progress-bar {
    background: linear-gradient(90deg, #10B981, #059669);
}

.achievement-notification {
    animation: slideInRight 0.5s ease-out;
}

@keyframes slideInRight {
    from {
        transform: translateX(100%);
        opacity: 0;
    }
    to {
        transform: translateX(0);
        opacity: 1;
    }
}

.level-up-modal .bg-white {
    animation: levelUpBounce 0.6s ease-out;
}

@keyframes levelUpBounce {
    0% {
        transform: scale(0.3) rotate(-10deg);
        opacity: 0;
    }
    50% {
        transform: scale(1.05) rotate(5deg);
    }
    100% {
        transform: scale(1) rotate(0deg);
        opacity: 1;
    }
}

.health-score-glow {
    box-shadow: 0 0 20px rgba(16, 185, 129, 0.4);
    animation: healthGlow 3s ease-in-out infinite;
}

@keyframes healthGlow {
    0%, 100% {
        box-shadow: 0 0 20px rgba(16, 185, 129, 0.4);
    }
    50% {
        box-shadow: 0 0 30px rgba(16, 185, 129, 0.6);
    }
}

.points-celebration {
    animation: pointsCelebration 1s ease-out;
}

@keyframes pointsCelebration {
    0% {
        transform: scale(1);
    }
    50% {
        transform: scale(1.2);
        color: #10B981;
    }
    100% {
        transform: scale(1);
    }
}
</style>
`;

// Inject styles
document.head.insertAdjacentHTML('beforeend', gamificationStyles);

// Export for global access
window.GamificationUI = GamificationUI;
window.ParticleSystem = ParticleSystem;
window.AchievementTemplates = AchievementTemplates;