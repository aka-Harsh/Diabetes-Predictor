// Enhanced Diabetes Predictor - Main JavaScript
class HealthPredict {
    constructor() {
        this.apiClient = new APIClient();
        this.currentUser = null;
        this.healthTasks = null;
        this.theme = this.getStoredTheme() || 'light';
    }

    // Helper method to safely access localStorage
    getStoredTheme() {
        try {
            return localStorage.getItem('theme');
        } catch (error) {
            console.warn('localStorage not available:', error);
            return null;
        }
    }

    // Helper method to safely set localStorage
    setStoredTheme(theme) {
        try {
            localStorage.setItem('theme', theme);
        } catch (error) {
            console.warn('Failed to save theme to localStorage:', error);
        }
    }

    static init() {
        const app = new HealthPredict();
        window.HealthPredict = app;
        app.initialize();
        return app;
    }

    initialize() {
        // Check for saved user preferences
        this.loadUserPreferences();
        
        // Set up event listeners
        this.setupEventListeners();
        
        // Initialize components
        this.initializeComponents();
        
        // Apply theme
        this.applyTheme();
        
        console.log('HealthPredict initialized successfully');
    }

    loadUserPreferences() {
        const theme = this.getStoredTheme();
        if (theme === 'dark') {
            document.documentElement.classList.add('dark');
        }
    }

    setupEventListeners() {
        // Global error handling
        window.addEventListener('error', (event) => {
            console.error('Application error:', event.error);
            this.showToast('An unexpected error occurred', 'error');
        });

        // Handle online/offline status
        window.addEventListener('online', () => {
            this.showToast('Connection restored', 'success');
        });

        window.addEventListener('offline', () => {
            this.showToast('Connection lost - some features may be limited', 'warning');
        });

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.altKey && e.key === 'h') {
                e.preventDefault();
                this.showHealthTasks();
            } else if (e.altKey && e.key === 'c') {
                e.preventDefault();
                this.showAIChat();
            }
        });
    }

    initializeComponents() {
        // Initialize any page-specific components
        if (typeof FormWizard !== 'undefined' && document.querySelector('.form-wizard')) {
            this.formWizard = new FormWizard();
        }
        
        if (typeof GamificationUI !== 'undefined') {
            this.gamificationUI = new GamificationUI();
        }

        // Initialize tooltips
        this.initializeTooltips();
    }

    initializeTooltips() {
        const tooltipElements = document.querySelectorAll('[data-tooltip]');
        tooltipElements.forEach(element => {
            element.addEventListener('mouseenter', this.showTooltip.bind(this));
            element.addEventListener('mouseleave', this.hideTooltip.bind(this));
        });
    }

    showTooltip(event) {
        const element = event.target;
        const text = element.getAttribute('data-tooltip');
        
        if (!text) return;

        const tooltip = document.createElement('div');
        tooltip.className = 'tooltip absolute z-50 bg-gray-800 text-white text-sm px-2 py-1 rounded shadow-lg';
        tooltip.textContent = text;
        tooltip.id = 'active-tooltip';

        document.body.appendChild(tooltip);

        const rect = element.getBoundingClientRect();
        tooltip.style.left = rect.left + (rect.width / 2) - (tooltip.offsetWidth / 2) + 'px';
        tooltip.style.top = rect.top - tooltip.offsetHeight - 5 + 'px';
    }

    hideTooltip() {
        const tooltip = document.getElementById('active-tooltip');
        if (tooltip) {
            tooltip.remove();
        }
    }

    applyTheme() {
        document.documentElement.setAttribute('data-theme', this.theme);
        if (this.theme === 'dark') {
            document.documentElement.classList.add('dark');
        } else {
            document.documentElement.classList.remove('dark');
        }
    }

    toggleTheme() {
        this.theme = this.theme === 'light' ? 'dark' : 'light';
        this.setStoredTheme(this.theme);
        this.applyTheme();
        this.showToast(`Switched to ${this.theme} mode`, 'info');
    }

    // Added missing method reference
    showHealthTasks() {
        showHealthTasks();
    }

    showAIChat() {
        showAIChat();
    }

    showToast(message, type = 'info', duration = 5000) {
        const container = document.getElementById('toast-container');
        if (!container) return;

        const toast = document.createElement('div');
        toast.className = `
            transform transition-all duration-300 ease-in-out translate-x-full
            bg-white border-l-4 rounded-lg shadow-lg p-4 mb-2 max-w-sm
            ${type === 'success' ? 'border-green-400' : ''}
            ${type === 'error' ? 'border-red-400' : ''}
            ${type === 'warning' ? 'border-yellow-400' : ''}
            ${type === 'info' ? 'border-blue-400' : ''}
            dark:bg-gray-800 dark:text-white
        `;

        const icons = {
            success: '✅',
            error: '❌',
            warning: '⚠️',
            info: 'ℹ️'
        };

        const textColors = {
            success: 'text-green-800 dark:text-green-300',
            error: 'text-red-800 dark:text-red-300',
            warning: 'text-yellow-800 dark:text-yellow-300',
            info: 'text-blue-800 dark:text-blue-300'
        };

        const icon = icons[type] || 'ℹ️';
        const textColor = textColors[type] || 'text-blue-800 dark:text-blue-300';

        toast.innerHTML = `
            <div class="flex">
                <div class="flex-shrink-0">
                    <span class="text-lg">${icon}</span>
                </div>
                <div class="ml-3">
                    <p class="text-sm font-medium ${textColor}">${message}</p>
                </div>
                <div class="ml-auto pl-3">
                    <button class="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300" onclick="this.parentElement.parentElement.parentElement.remove()">
                        <span class="text-lg">×</span>
                    </button>
                </div>
            </div>
        `;

        container.appendChild(toast);

        // Animate in
        setTimeout(() => {
            toast.classList.remove('translate-x-full');
        }, 100);

        // Auto remove
        setTimeout(() => {
            toast.classList.add('translate-x-full');
            setTimeout(() => {
                if (toast.parentElement) {
                    toast.remove();
                }
            }, 300);
        }, duration);

        // Add sound notification
        this.playNotificationSound(type);
    }

    playNotificationSound(type) {
        try {
            const audioContext = new (window.AudioContext || window.webkitAudioContext)();
            const oscillator = audioContext.createOscillator();
            const gainNode = audioContext.createGain();

            oscillator.connect(gainNode);
            gainNode.connect(audioContext.destination);

            const frequencies = {
                success: 523.25, // C5
                error: 311.13,   // Eb4
                warning: 415.30, // Ab4
                info: 440.00     // A4
            };

            oscillator.frequency.value = frequencies[type] || frequencies.info;
            oscillator.type = 'sine';

            gainNode.gain.setValueAtTime(0.1, audioContext.currentTime);
            gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.3);

            oscillator.start(audioContext.currentTime);
            oscillator.stop(audioContext.currentTime + 0.3);
        } catch (error) {
            // Audio not supported or blocked
            console.log('Audio notification not available');
        }
    }
}

// API Client for backend communication
class APIClient {
    constructor() {
        this.baseURL = window.location.origin;
        this.defaultHeaders = {
            'Content-Type': 'application/json',
            'X-Requested-With': 'XMLHttpRequest'
        };
    }

    async request(endpoint, options = {}) {
        const url = `${this.baseURL}/api${endpoint}`;
        
        const config = {
            headers: { ...this.defaultHeaders, ...options.headers },
            ...options
        };

        try {
            showLoading(true);
            const response = await fetch(url, config);
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const data = await response.json();
            return data;
        } catch (error) {
            console.error('API Request failed:', error);
            throw error;
        } finally {
            showLoading(false);
        }
    }

    async get(endpoint, params = {}) {
        const queryString = new URLSearchParams(params).toString();
        const url = queryString ? `${endpoint}?${queryString}` : endpoint;
        
        return this.request(url, { method: 'GET' });
    }

    async post(endpoint, data = {}) {
        return this.request(endpoint, {
            method: 'POST',
            body: JSON.stringify(data)
        });
    }

    async put(endpoint, data = {}) {
        return this.request(endpoint, {
            method: 'PUT',
            body: JSON.stringify(data)
        });
    }

    async delete(endpoint) {
        return this.request(endpoint, { method: 'DELETE' });
    }

    async predict(data) {
        return this.post('/predict', data);
    }

    async updateHealthTask(taskType, completed, value) {
        return this.post('/health-tasks', {
            task_type: taskType,
            completed: completed,
            value: value
        });
    }

    async getHealthTasks() {
        return this.get('/health-tasks');
    }

    async getChatResponse(message) {
        return this.post('/ai-chat', { message: message });
    }

    async getScenarios(baseData, modifications) {
        return this.post('/scenarios', {
            base_data: baseData,
            modifications: modifications
        });
    }

    async getPopulationStats() {
        return this.get('/population-stats');
    }

    async exportData() {
        return this.get('/export-data');
    }
}

// Utility Functions
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function formatNumber(num, decimals = 1) {
    return Number(num).toFixed(decimals);
}

function formatPercentage(num) {
    return Math.round(num * 100) + '%';
}

function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

function throttle(func, limit) {
    let inThrottle;
    return function() {
        const args = arguments;
        const context = this;
        if (!inThrottle) {
            func.apply(context, args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    };
}

// Local Storage Helpers
function saveToLocalStorage(key, data) {
    try {
        localStorage.setItem(key, JSON.stringify(data));
    } catch (error) {
        console.warn('Failed to save to localStorage:', error);
    }
}

function loadFromLocalStorage(key, defaultValue = null) {
    try {
        const item = localStorage.getItem(key);
        return item ? JSON.parse(item) : defaultValue;
    } catch (error) {
        console.warn('Failed to load from localStorage:', error);
        return defaultValue;
    }
}

// Data Export Functions
async function exportUserData() {
    try {
        showLoading(true);
        const api = new APIClient();
        const data = await api.exportData();
        
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        
        const link = document.createElement('a');
        link.href = url;
        link.download = `health_data_${new Date().toISOString().split('T')[0]}.json`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
        
        showToast('Health data exported successfully', 'success');
    } catch (error) {
        console.error('Export error:', error);
        showToast('Failed to export data', 'error');
    } finally {
        showLoading(false);
    }
}

// Keyboard Shortcuts
function setupKeyboardShortcuts() {
    document.addEventListener('keydown', (e) => {
        // Alt + H: Show Health Tasks
        if (e.altKey && e.key === 'h') {
            e.preventDefault();
            showHealthTasks();
        }
        
        // Alt + C: Show Chat
        if (e.altKey && e.key === 'c') {
            e.preventDefault();
            showAIChat();
        }
        
        // Alt + T: Toggle Theme
        if (e.altKey && e.key === 't') {
            e.preventDefault();
            if (window.HealthPredict) {
                window.HealthPredict.toggleTheme();
            }
        }
        
        // Escape: Close modals
        if (e.key === 'Escape') {
            const openModals = document.querySelectorAll('.modal:not(.hidden)');
            openModals.forEach(modal => {
                closeModal(modal.id);
            });
        }
    });
}

// Animation Helpers
function animateValue(element, start, end, duration = 1000) {
    const range = end - start;
    const increment = range / (duration / 16);
    let current = start;
    
    const timer = setInterval(() => {
        current += increment;
        element.textContent = Math.round(current);
        
        if ((increment > 0 && current >= end) || (increment < 0 && current <= end)) {
            element.textContent = end;
            clearInterval(timer);
        }
    }, 16);
}

function pulse(element, duration = 200) {
    element.style.transform = 'scale(1.05)';
    element.style.transition = `transform ${duration}ms ease`;
    
    setTimeout(() => {
        element.style.transform = 'scale(1)';
    }, duration);
}

// Accessibility Helpers
function announceToScreenReader(message) {
    const announcement = document.createElement('div');
    announcement.setAttribute('aria-live', 'polite');
    announcement.setAttribute('aria-atomic', 'true');
    announcement.className = 'sr-only';
    announcement.textContent = message;
    
    document.body.appendChild(announcement);
    
    setTimeout(() => {
        document.body.removeChild(announcement);
    }, 1000);
}

// Performance Monitoring
function measurePerformance(name, fn) {
    const start = performance.now();
    const result = fn();
    const end = performance.now();
    
    console.log(`${name} took ${end - start} milliseconds`);
    return result;
}

// Export for global access
window.showLoading = showLoading;
window.showToast = (message, type, duration) => {
    if (window.HealthPredict) {
        window.HealthPredict.showToast(message, type, duration);
    }
};
window.showModal = showModal;
window.closeModal = closeModal;
window.showHealthTasks = showHealthTasks;
window.updateTask = updateTask;
window.refreshHealthTasks = refreshHealthTasks;
window.showAIChat = showAIChat;
window.sendChatMessage = sendChatMessage;
window.clearChatHistory = clearChatHistory;
window.toggleTheme = () => {
    if (window.HealthPredict) {
        window.HealthPredict.toggleTheme();
    }
};
window.exportUserData = exportUserData;
window.escapeHtml = escapeHtml;
window.formatNumber = formatNumber;
window.formatPercentage = formatPercentage;
window.debounce = debounce;
window.throttle = throttle;

// Initialize when DOM is loaded
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        HealthPredict.init();
        setupKeyboardShortcuts();
    });
} else {
    HealthPredict.init();
    setupKeyboardShortcuts();
} 

function showLoading(show = true) {
    const overlay = document.getElementById('loading-overlay');
    if (overlay) {
        if (show) {
            overlay.classList.remove('hidden');
            overlay.setAttribute('aria-hidden', 'false');
        } else {
            overlay.classList.add('hidden');
            overlay.setAttribute('aria-hidden', 'true');
        }
    }
}

function showModal(modalId) {
    const modal = document.getElementById(modalId);
    if (modal) {
        modal.classList.remove('hidden');
        modal.setAttribute('aria-hidden', 'false');
        document.body.classList.add('overflow-hidden');
        
        // Focus management for accessibility
        const focusableElements = modal.querySelectorAll('button, input, textarea, select, [tabindex]:not([tabindex="-1"])');
        if (focusableElements.length > 0) {
            focusableElements[0].focus();
        }

        // Trap focus within modal
        modal.addEventListener('keydown', trapFocus);
    }
}

function closeModal(modalId) {
    const modal = document.getElementById(modalId);
    if (modal) {
        modal.classList.add('hidden');
        modal.setAttribute('aria-hidden', 'true');
        document.body.classList.remove('overflow-hidden');
        modal.removeEventListener('keydown', trapFocus);
    }
}

function trapFocus(event) {
    const modal = event.currentTarget;
    const focusableElements = modal.querySelectorAll('button, input, textarea, select, [tabindex]:not([tabindex="-1"])');
    const firstElement = focusableElements[0];
    const lastElement = focusableElements[focusableElements.length - 1];

    if (event.key === 'Tab') {
        if (event.shiftKey) {
            if (document.activeElement === firstElement) {
                event.preventDefault();
                lastElement.focus();
            }
        } else {
            if (document.activeElement === lastElement) {
                event.preventDefault();
                firstElement.focus();
            }
        }
    }

    if (event.key === 'Escape') {
        const modalId = modal.id;
        closeModal(modalId);
    }
}

// Health Tasks Modal Functions
async function showHealthTasks() {
    try {
        const modal = document.getElementById('health-tasks-modal');
        const content = document.getElementById('health-tasks-content');
        
        if (!content) {
            console.warn('Health tasks content container not found');
            return;
        }
        
        // Show modal first
        showModal('health-tasks-modal');
        
        // Load tasks
        content.innerHTML = '<div class="text-center py-8"><div class="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto"></div><p class="mt-4 text-gray-600">Loading health tasks...</p></div>';
        
        const api = new APIClient();
        const tasks = await api.getHealthTasks();
        
        // Render tasks
        renderHealthTasks(tasks, content);
        
    } catch (error) {
        console.error('Error loading health tasks:', error);
        showToast('Failed to load health tasks', 'error');
    }
}

function renderHealthTasks(tasks, container) {
    if (!tasks.tasks || Object.keys(tasks.tasks).length === 0) {
        container.innerHTML = `
            <div class="text-center py-8">
                <div class="text-4xl mb-4">📋</div>
                <p class="text-gray-600">No health tasks available</p>
                <button onclick="location.reload()" class="mt-4 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors duration-200">
                    Refresh
                </button>
            </div>
        `;
        return;
    }

    const tasksArray = Object.values(tasks.tasks);
    const completedCount = tasksArray.filter(task => task.completed).length;
    const completionRate = Math.round((completedCount / tasksArray.length) * 100);

    container.innerHTML = `
        <!-- Progress Overview -->
        <div class="bg-gradient-to-r from-blue-50 to-green-50 rounded-xl p-6 mb-6">
            <div class="flex justify-between items-center mb-4">
                <h4 class="text-lg font-semibold text-gray-900">Today's Progress</h4>
                <span class="text-2xl font-bold text-blue-600">${completionRate}%</span>
            </div>
            <div class="w-full bg-gray-200 rounded-full h-3">
                <div class="bg-gradient-to-r from-blue-500 to-green-500 h-3 rounded-full transition-all duration-500" 
                     style="width: ${completionRate}%"></div>
            </div>
            <div class="mt-2 flex justify-between text-sm text-gray-600">
                <span>${completedCount} of ${tasksArray.length} completed</span>
                <span>${tasks.earned_points} / ${tasks.total_possible_points} points</span>
            </div>
        </div>

        <!-- Tasks Grid -->
        <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
            ${tasksArray.map(task => renderTaskCard(task)).join('')}
        </div>
        
        <!-- Action Buttons -->
        <div class="mt-6 flex justify-center space-x-4">
            <button onclick="refreshHealthTasks()" class="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors duration-200">
                Refresh Tasks
            </button>
            <button onclick="closeModal('health-tasks-modal')" class="px-4 py-2 bg-gray-300 text-gray-700 rounded-lg hover:bg-gray-400 transition-colors duration-200">
                Close
            </button>
        </div>
    `;
}

function renderTaskCard(task) {
    const isCompleted = task.completed;
    const progress = task.target > 0 ? Math.min((task.current_value / task.target) * 100, 100) : 0;
    
    return `
        <div class="bg-white border border-gray-200 rounded-xl p-4 ${isCompleted ? 'ring-2 ring-green-500 bg-green-50' : ''} transition-all duration-200 hover:shadow-md">
            <div class="flex items-center justify-between mb-3">
                <div class="flex items-center">
                    <span class="text-2xl mr-3">${task.icon}</span>
                    <div>
                        <h5 class="font-semibold text-gray-900">${task.display_name}</h5>
                        <p class="text-sm text-gray-600">Target: ${task.target} ${task.unit}</p>
                    </div>
                </div>
                <div class="text-right">
                    <div class="text-lg font-bold ${isCompleted ? 'text-green-600' : 'text-gray-600'}">
                        ${task.points_earned}/${task.points}
                    </div>
                    <div class="text-xs text-gray-500">points</div>
                </div>
            </div>
            
            <!-- Progress Bar -->
            <div class="mb-3">
                <div class="w-full bg-gray-200 rounded-full h-2">
                    <div class="bg-gradient-to-r from-blue-500 to-green-500 h-2 rounded-full transition-all duration-500 ${isCompleted ? 'animate-pulse' : ''}" 
                         style="width: ${progress}%"></div>
                </div>
                <div class="mt-1 flex justify-between text-xs text-gray-600">
                    <span>Current: ${task.current_value} ${task.unit}</span>
                    <span>${Math.round(progress)}%</span>
                </div>
            </div>
            
            <!-- Task Controls -->
            <div class="flex items-center gap-2">
                <input type="number" 
                       id="task-${task.name}" 
                       value="${task.current_value}" 
                       min="0" 
                       max="${task.target * 3}"
                       step="0.1"
                       class="flex-1 px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-200"
                       placeholder="Enter ${task.unit}">
                <button onclick="updateTask('${task.name}', document.getElementById('task-${task.name}').value)"
                        class="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 transition-all duration-200 text-sm font-medium">
                    Update
                </button>
            </div>
            
            ${isCompleted ? '<div class="mt-2 text-center text-green-600 text-sm font-medium">✨ Completed! Great job!</div>' : ''}
        </div>
    `;
}

async function updateTask(taskName, value) {
    try {
        const numValue = parseFloat(value) || 0;
        const api = new APIClient();
        
        // Get task config to determine completion
        const tasksData = await api.getHealthTasks();
        const task = tasksData.tasks[taskName];
        
        if (!task) {
            showToast('Invalid task', 'error');
            return;
        }
        
        const completed = numValue >= task.target;
        
        const result = await api.updateHealthTask(taskName, completed, numValue);
        
        if (result.success) {
            showToast(`${task.display_name} updated! +${result.points_earned} points`, 'success');
            
            // Check for new achievements
            if (result.new_achievements && result.new_achievements.length > 0) {
                result.new_achievements.forEach(achievement => {
                    setTimeout(() => {
                        showAchievementNotification(achievement);
                    }, 1000);
                });
            }
            
            // Refresh the health tasks display
            setTimeout(() => showHealthTasks(), 1500);
        } else {
            showToast(result.error || 'Failed to update task', 'error');
        }
        
    } catch (error) {
        console.error('Error updating task:', error);
        showToast('Failed to update task', 'error');
    }
}

async function refreshHealthTasks() {
    try {
        showHealthTasks();
        showToast('Health tasks refreshed', 'info');
    } catch (error) {
        console.error('Error refreshing tasks:', error);
        showToast('Failed to refresh tasks', 'error');
    }
}

function showAchievementNotification(achievement) {
    const notification = document.createElement('div');
    notification.className = 'achievement-notification fixed top-4 right-4 z-50';
    
    notification.innerHTML = `
        <div class="bg-gradient-to-r from-yellow-400 to-orange-400 text-yellow-900 p-4 rounded-lg shadow-lg transform transition-all duration-500 translate-x-full max-w-sm">
            <div class="flex items-center">
                <span class="text-3xl mr-3 animate-bounce">${achievement.icon}</span>
                <div>
                    <div class="font-bold text-sm">🎉 Achievement Unlocked!</div>
                    <div class="font-semibold">${achievement.title}</div>
                    <div class="text-sm opacity-90">${achievement.description}</div>
                    <div class="text-xs mt-1">+${achievement.points} points earned!</div>
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
    }, 6000);

    // Play celebration sound
    playCelebrationSound();
}

function playCelebrationSound() {
    try {
        const audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const notes = [523.25, 659.25, 783.99, 1046.50]; // C5, E5, G5, C6
        
        notes.forEach((freq, index) => {
            setTimeout(() => {
                const oscillator = audioContext.createOscillator();
                const gainNode = audioContext.createGain();
                
                oscillator.connect(gainNode);
                gainNode.connect(audioContext.destination);
                
                oscillator.frequency.value = freq;
                oscillator.type = 'sine';
                
                gainNode.gain.setValueAtTime(0.1, audioContext.currentTime);
                gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.4);
                
                oscillator.start(audioContext.currentTime);
                oscillator.stop(audioContext.currentTime + 0.4);
            }, index * 150);
        });
    } catch (error) {
        console.log('Celebration sound not available');
    }
}

// AI Chat Functions
async function showAIChat() {
    showModal('ai-chat-modal');
    
    // Focus on input
    const chatInput = document.getElementById('chat-input');
    if (chatInput) {
        chatInput.focus();
        
        // Add enter key listener (only once)
        if (!chatInput.hasAttribute('data-listener-added')) {
            chatInput.addEventListener('keypress', function(e) {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    sendChatMessage();
                }
            });
            chatInput.setAttribute('data-listener-added', 'true');
        }
    }

    // Initialize chat if empty
    const messagesContainer = document.getElementById('chat-messages');
    if (messagesContainer && messagesContainer.children.length === 0) {
        addChatMessage("Hello! I'm your AI health assistant. I can help you understand your diabetes risk, provide health guidance, and answer questions about your assessment results. What would you like to know?", 'ai');
    }
}

async function sendChatMessage() {
    const input = document.getElementById('chat-input');
    const messagesContainer = document.getElementById('chat-messages');
    
    if (!input || !messagesContainer) return;
    
    const message = input.value.trim();
    if (!message) return;
    
    // Clear input
    input.value = '';
    
    // Add user message
    addChatMessage(message, 'user');
    
    // Show typing indicator
    const typingId = addTypingIndicator();
    
    try {
        const api = new APIClient();
        const response = await api.getChatResponse(message);
        
        // Remove typing indicator
        removeTypingIndicator(typingId);
        
        // Add AI response
        addChatMessage(response.response || 'Sorry, I encountered an error. Please try again.', 'ai');
        
    } catch (error) {
        console.error('Chat error:', error);
        removeTypingIndicator(typingId);
        addChatMessage('I\'m sorry, I\'m having trouble connecting right now. Please try again later.', 'ai');
        showToast('AI assistant temporarily unavailable', 'warning');
    }
    
    // Scroll to bottom
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

function addChatMessage(message, sender) {
    const messagesContainer = document.getElementById('chat-messages');
    if (!messagesContainer) return;
    
    const messageDiv = document.createElement('div');
    messageDiv.className = sender === 'user' ? 'user-message' : 'ai-message';
    
    const timestamp = new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
    
    if (sender === 'user') {
        messageDiv.innerHTML = `
            <div class="flex justify-end mb-4">
                <div class="bg-blue-600 text-white p-3 rounded-lg max-w-xs lg:max-w-md">
                    <div class="text-sm">${escapeHtml(message)}</div>
                    <div class="text-xs opacity-75 mt-1">${timestamp}</div>
                </div>
            </div>
        `;
    } else {
        messageDiv.innerHTML = `
            <div class="flex mb-4">
                <div class="bg-gray-100 dark:bg-gray-700 text-gray-800 dark:text-gray-200 p-3 rounded-lg max-w-xs lg:max-w-md">
                    <div class="text-sm">${escapeHtml(message)}</div>
                    <div class="text-xs opacity-75 mt-1">${timestamp}</div>
                </div>
            </div>
        `;
    }
    
    messagesContainer.appendChild(messageDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

function addTypingIndicator() {
    const messagesContainer = document.getElementById('chat-messages');
    if (!messagesContainer) return null;
    
    const typingDiv = document.createElement('div');
    const typingId = 'typing-' + Date.now();
    typingDiv.id = typingId;
    typingDiv.className = 'ai-message';
    typingDiv.innerHTML = `
        <div class="flex mb-4">
            <div class="bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-400 p-3 rounded-lg">
                <div class="flex items-center space-x-1">
                    <div class="flex space-x-1">
                        <div class="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                        <div class="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style="animation-delay: 0.1s"></div>
                        <div class="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style="animation-delay: 0.2s"></div>
                    </div>
                    <span class="text-sm ml-2">AI is typing...</span>
                </div>
            </div>
        </div>
    `;
    
    messagesContainer.appendChild(typingDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
    
    return typingId;
}

function removeTypingIndicator(typingId) {
    if (typingId) {
        const typingDiv = document.getElementById(typingId);
        if (typingDiv) {
            typingDiv.remove();
        }
    }
}

function clearChatHistory() {
    const messagesContainer = document.getElementById('chat-messages');
    if (messagesContainer) {
        messagesContainer.innerHTML = `
            <div class="ai-message">
                <div class="flex mb-4">
                    <div class="bg-blue-100 text-blue-800 p-3 rounded-lg">
                        Hello! I'm your AI health assistant. How can I help you today?
                    </div>
                </div>
            </div>
        `;
    }
}
