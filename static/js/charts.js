// Enhanced Diabetes Predictor - Charts and Data Visualization
class ChartRenderer {
    constructor() {
        this.charts = new Map();
        this.defaultOptions = {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top',
                    labels: {
                        boxWidth: 12,
                        font: {
                            size: 12
                        }
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(255, 255, 255, 0.95)',
                    titleColor: '#374151',
                    bodyColor: '#6B7280',
                    borderColor: '#E5E7EB',
                    borderWidth: 1,
                    cornerRadius: 8,
                    displayColors: true,
                    titleFont: {
                        size: 14,
                        weight: 'bold'
                    },
                    bodyFont: {
                        size: 12
                    }
                }
            },
            elements: {
                point: {
                    radius: 4,
                    hoverRadius: 6,
                },
                line: {
                    borderWidth: 3,
                    tension: 0.4,
                }
            },
            animation: {
                duration: 1000,
                easing: 'easeInOutQuart'
            }
        };
    }

    // Risk Trend Chart
    createRiskTrendChart(canvasId, data) {
        const canvas = document.getElementById(canvasId);
        if (!canvas) {
            console.warn(`Canvas with id '${canvasId}' not found`);
            return null;
        }

        const ctx = canvas.getContext('2d');
        
        // Destroy existing chart
        if (this.charts.has(canvasId)) {
            this.charts.get(canvasId).destroy();
        }

        const chartData = {
            labels: data.dates || [],
            datasets: [{
                label: 'Diabetes Risk %',
                data: data.riskScores || [],
                borderColor: '#EF4444',
                backgroundColor: 'rgba(239, 68, 68, 0.1)',
                fill: true,
                tension: 0.4,
                pointBackgroundColor: '#EF4444',
                pointBorderColor: '#ffffff',
                pointBorderWidth: 2
            }, {
                label: 'Health Score %',
                data: data.healthScores || [],
                borderColor: '#10B981',
                backgroundColor: 'rgba(16, 185, 129, 0.1)',
                fill: true,
                tension: 0.4,
                pointBackgroundColor: '#10B981',
                pointBorderColor: '#ffffff',
                pointBorderWidth: 2
            }]
        };

        const options = {
            ...this.defaultOptions,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    ticks: {
                        callback: function(value) {
                            return value + '%';
                        },
                        font: {
                            size: 11
                        }
                    },
                    grid: {
                        color: 'rgba(156, 163, 175, 0.2)',
                        drawBorder: false
                    }
                },
                x: {
                    grid: {
                        color: 'rgba(156, 163, 175, 0.2)',
                        drawBorder: false
                    },
                    ticks: {
                        font: {
                            size: 11
                        }
                    }
                }
            },
            plugins: {
                ...this.defaultOptions.plugins,
                title: {
                    display: true,
                    text: 'Health Trend Over Time',
                    font: {
                        size: 16,
                        weight: 'bold'
                    }
                }
            }
        };

        const chart = new Chart(ctx, {
            type: 'line',
            data: chartData,
            options: options
        });

        this.charts.set(canvasId, chart);
        return chart;
    }

    // Risk Factors Radar Chart
    createRiskFactorsRadarChart(canvasId, data) {
        const canvas = document.getElementById(canvasId);
        if (!canvas) {
            console.warn(`Canvas with id '${canvasId}' not found`);
            return null;
        }

        const ctx = canvas.getContext('2d');
        
        if (this.charts.has(canvasId)) {
            this.charts.get(canvasId).destroy();
        }

        const chartData = {
            labels: data.factors || ['Age', 'BMI', 'Blood Glucose', 'HbA1c', 'Blood Pressure', 'Lifestyle'],
            datasets: [{
                label: 'Risk Level',
                data: data.values || [0, 0, 0, 0, 0, 0],
                borderColor: '#EF4444',
                backgroundColor: 'rgba(239, 68, 68, 0.2)',
                borderWidth: 2,
                pointBackgroundColor: '#EF4444',
                pointBorderColor: '#ffffff',
                pointHoverBackgroundColor: '#ffffff',
                pointHoverBorderColor: '#EF4444',
                pointRadius: 4,
                pointHoverRadius: 6
            }]
        };

        const options = {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                r: {
                    angleLines: {
                        color: 'rgba(156, 163, 175, 0.3)'
                    },
                    grid: {
                        color: 'rgba(156, 163, 175, 0.3)'
                    },
                    pointLabels: {
                        font: {
                            size: 12,
                            weight: '500'
                        },
                        color: '#374151'
                    },
                    ticks: {
                        display: false
                    },
                    suggestedMin: 0,
                    suggestedMax: 1
                }
            },
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `${context.label}: ${(context.parsed.r * 100).toFixed(1)}%`;
                        }
                    }
                },
                title: {
                    display: true,
                    text: 'Risk Factors Analysis',
                    font: {
                        size: 16,
                        weight: 'bold'
                    }
                }
            },
            animation: {
                duration: 1500
            }
        };

        const chart = new Chart(ctx, {
            type: 'radar',
            data: chartData,
            options: options
        });

        this.charts.set(canvasId, chart);
        return chart;
    }

    // Health Tasks Completion Chart
    createTaskCompletionChart(canvasId, data) {
        const canvas = document.getElementById(canvasId);
        if (!canvas) {
            console.warn(`Canvas with id '${canvasId}' not found`);
            return null;
        }

        const ctx = canvas.getContext('2d');
        
        if (this.charts.has(canvasId)) {
            this.charts.get(canvasId).destroy();
        }

        const chartData = {
            labels: data.taskNames || [],
            datasets: [{
                data: data.completionRates || [],
                backgroundColor: [
                    '#3B82F6', // Blue
                    '#10B981', // Green
                    '#F59E0B', // Yellow
                    '#EF4444', // Red
                    '#8B5CF6', // Purple
                    '#F97316'  // Orange
                ],
                borderColor: '#ffffff',
                borderWidth: 2,
                hoverBorderWidth: 3,
                hoverOffset: 4
            }]
        };

        const options = {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'right',
                    labels: {
                        boxWidth: 12,
                        font: {
                            size: 11
                        },
                        padding: 15,
                        usePointStyle: true
                    }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `${context.label}: ${context.parsed}% completed`;
                        }
                    }
                },
                title: {
                    display: true,
                    text: 'Task Completion Rate',
                    font: {
                        size: 16,
                        weight: 'bold'
                    }
                }
            },
            animation: {
                animateRotate: true,
                animateScale: true,
                duration: 1000
            }
        };

        const chart = new Chart(ctx, {
            type: 'doughnut',
            data: chartData,
            options: options
        });

        this.charts.set(canvasId, chart);
        return chart;
    }

    // BMI Distribution Chart
    createBMIDistributionChart(canvasId, data) {
        const canvas = document.getElementById(canvasId);
        if (!canvas) {
            console.warn(`Canvas with id '${canvasId}' not found`);
            return null;
        }

        const ctx = canvas.getContext('2d');
        
        if (this.charts.has(canvasId)) {
            this.charts.get(canvasId).destroy();
        }

        const chartData = {
            labels: ['Underweight', 'Normal', 'Overweight', 'Obese'],
            datasets: [{
                label: 'Population %',
                data: data.distribution || [5, 45, 35, 15],
                backgroundColor: [
                    'rgba(59, 130, 246, 0.8)',
                    'rgba(16, 185, 129, 0.8)',
                    'rgba(245, 158, 11, 0.8)',
                    'rgba(239, 68, 68, 0.8)'
                ],
                borderColor: [
                    '#3B82F6',
                    '#10B981',
                    '#F59E0B',
                    '#EF4444'
                ],
                borderWidth: 2,
                borderRadius: 4
            }, {
                label: 'Your Category',
                data: data.userCategory || [0, 0, 0, 0],
                backgroundColor: 'rgba(99, 102, 241, 0.8)',
                borderColor: '#6366F1',
                borderWidth: 3,
                type: 'bar'
            }]
        };

        const options = {
            ...this.defaultOptions,
            plugins: {
                ...this.defaultOptions.plugins,
                title: {
                    display: true,
                    text: 'BMI Distribution Comparison',
                    font: {
                        size: 16,
                        weight: 'bold'
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: {
                        callback: function(value) {
                            return value + '%';
                        },
                        font: {
                            size: 11
                        }
                    },
                    grid: {
                        color: 'rgba(156, 163, 175, 0.2)'
                    }
                },
                x: {
                    grid: {
                        display: false
                    },
                    ticks: {
                        font: {
                            size: 11
                        }
                    }
                }
            }
        };

        const chart = new Chart(ctx, {
            type: 'bar',
            data: chartData,
            options: options
        });

        this.charts.set(canvasId, chart);
        return chart;
    }

    // Weekly Progress Chart
    createWeeklyProgressChart(canvasId, data) {
        const canvas = document.getElementById(canvasId);
        if (!canvas) {
            console.warn(`Canvas with id '${canvasId}' not found`);
            return null;
        }

        const ctx = canvas.getContext('2d');
        
        if (this.charts.has(canvasId)) {
            this.charts.get(canvasId).destroy();
        }

        const chartData = {
            labels: data.days || ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
            datasets: [{
                label: 'Tasks Completed',
                data: data.completedTasks || [0, 0, 0, 0, 0, 0, 0],
                backgroundColor: 'rgba(16, 185, 129, 0.8)',
                borderColor: '#10B981',
                borderWidth: 2,
                borderRadius: 4,
                borderSkipped: false,
                maxBarThickness: 40
            }, {
                label: 'Total Tasks',
                data: data.totalTasks || [6, 6, 6, 6, 6, 6, 6],
                backgroundColor: 'rgba(156, 163, 175, 0.3)',
                borderColor: '#9CA3AF',
                borderWidth: 1,
                borderRadius: 4,
                borderSkipped: false,
                maxBarThickness: 40
            }]
        };

        const options = {
            ...this.defaultOptions,
            plugins: {
                ...this.defaultOptions.plugins,
                title: {
                    display: true,
                    text: 'Weekly Task Completion',
                    font: {
                        size: 16,
                        weight: 'bold'
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: {
                        stepSize: 1,
                        font: {
                            size: 11
                        }
                    },
                    grid: {
                        color: 'rgba(156, 163, 175, 0.2)'
                    }
                },
                x: {
                    grid: {
                        display: false
                    },
                    ticks: {
                        font: {
                            size: 11
                        }
                    }
                }
            },
            interaction: {
                intersect: false,
                mode: 'index'
            }
        };

        const chart = new Chart(ctx, {
            type: 'bar',
            data: chartData,
            options: options
        });

        this.charts.set(canvasId, chart);
        return chart;
    }

    // Model Performance Comparison Chart
    createModelComparisonChart(canvasId, data) {
        const canvas = document.getElementById(canvasId);
        if (!canvas) {
            console.warn(`Canvas with id '${canvasId}' not found`);
            return null;
        }

        const ctx = canvas.getContext('2d');
        
        if (this.charts.has(canvasId)) {
            this.charts.get(canvasId).destroy();
        }

        const chartData = {
            labels: data.models || ['Random Forest', 'XGBoost', 'Logistic Regression'],
            datasets: [{
                label: 'Risk Probability (%)',
                data: data.probabilities || [0, 0, 0],
                backgroundColor: [
                    'rgba(59, 130, 246, 0.8)',
                    'rgba(16, 185, 129, 0.8)',
                    'rgba(245, 158, 11, 0.8)'
                ],
                borderColor: [
                    '#3B82F6',
                    '#10B981',
                    '#F59E0B'
                ],
                borderWidth: 2,
                borderRadius: 8,
                maxBarThickness: 60
            }]
        };

        const options = {
            ...this.defaultOptions,
            indexAxis: 'y',
            plugins: {
                ...this.defaultOptions.plugins,
                title: {
                    display: true,
                    text: 'Model Predictions Comparison',
                    font: {
                        size: 16,
                        weight: 'bold'
                    }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `${context.dataset.label}: ${context.parsed.x.toFixed(1)}%`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    beginAtZero: true,
                    max: 100,
                    ticks: {
                        callback: function(value) {
                            return value + '%';
                        },
                        font: {
                            size: 11
                        }
                    },
                    grid: {
                        color: 'rgba(156, 163, 175, 0.2)'
                    }
                },
                y: {
                    grid: {
                        display: false
                    },
                    ticks: {
                        font: {
                            size: 11
                        }
                    }
                }
            }
        };

        const chart = new Chart(ctx, {
            type: 'bar',
            data: chartData,
            options: options
        });

        this.charts.set(canvasId, chart);
        return chart;
    }

    // Risk Over Time Chart
    createRiskOverTimeChart(canvasId, data) {
        const canvas = document.getElementById(canvasId);
        if (!canvas) {
            console.warn(`Canvas with id '${canvasId}' not found`);
            return null;
        }

        const ctx = canvas.getContext('2d');
        
        if (this.charts.has(canvasId)) {
            this.charts.get(canvasId).destroy();
        }

        const chartData = {
            labels: data.dates || [],
            datasets: [{
                label: 'Risk Probability',
                data: data.riskValues || [],
                borderColor: '#EF4444',
                backgroundColor: 'rgba(239, 68, 68, 0.1)',
                fill: true,
                tension: 0.4,
                pointBackgroundColor: '#EF4444',
                pointBorderColor: '#ffffff',
                pointBorderWidth: 2
            }]
        };

        const options = {
            ...this.defaultOptions,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 1,
                    ticks: {
                        callback: function(value) {
                            return (value * 100).toFixed(0) + '%';
                        },
                        font: {
                            size: 11
                        }
                    },
                    grid: {
                        color: 'rgba(156, 163, 175, 0.2)'
                    }
                },
                x: {
                    grid: {
                        color: 'rgba(156, 163, 175, 0.2)'
                    },
                    ticks: {
                        font: {
                            size: 11
                        }
                    }
                }
            },
            plugins: {
                ...this.defaultOptions.plugins,
                title: {
                    display: true,
                    text: 'Risk Trend Analysis',
                    font: {
                        size: 16,
                        weight: 'bold'
                    }
                }
            }
        };

        const chart = new Chart(ctx, {
            type: 'line',
            data: chartData,
            options: options
        });

        this.charts.set(canvasId, chart);
        return chart;
    }

    // Health Score Gauge Chart
    createHealthScoreGaugeChart(canvasId, score) {
        const canvas = document.getElementById(canvasId);
        if (!canvas) {
            console.warn(`Canvas with id '${canvasId}' not found`);
            return null;
        }

        const ctx = canvas.getContext('2d');
        
        if (this.charts.has(canvasId)) {
            this.charts.get(canvasId).destroy();
        }

        // Create gauge chart using doughnut chart
        const chartData = {
            datasets: [{
                data: [score, 100 - score],
                backgroundColor: [
                    this.getScoreColor(score),
                    'rgba(229, 231, 235, 0.3)'
                ],
                borderWidth: 0,
                circumference: 180,
                rotation: 270,
                cutout: '80%'
            }]
        };

        const options = {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    enabled: false
                }
            },
            animation: {
                animateRotate: true,
                duration: 2000
            }
        };

        const chart = new Chart(ctx, {
            type: 'doughnut',
            data: chartData,
            options: options,
            plugins: [{
                afterDraw: function(chart) {
                    const ctx = chart.ctx;
                    const centerX = chart.width / 2;
                    const centerY = chart.height / 2 + 20;
                    
                    ctx.fillStyle = '#374151';
                    ctx.font = 'bold 24px Inter, sans-serif';
                    ctx.textAlign = 'center';
                    ctx.fillText(Math.round(score), centerX, centerY);
                    
                    ctx.fillStyle = '#6B7280';
                    ctx.font = '12px Inter, sans-serif';
                    ctx.fillText('Health Score', centerX, centerY + 20);
                }
            }]
        });

        this.charts.set(canvasId, chart);
        return chart;
    }

    getScoreColor(score) {
        if (score >= 80) return '#10B981'; // Green
        if (score >= 60) return '#3B82F6'; // Blue
        if (score >= 40) return '#F59E0B'; // Yellow
        return '#EF4444'; // Red
    }

    // Destroy chart by ID
    destroyChart(canvasId) {
        if (this.charts.has(canvasId)) {
            this.charts.get(canvasId).destroy();
            this.charts.delete(canvasId);
        }
    }

    // Destroy all charts
    destroyAllCharts() {
        this.charts.forEach(chart => chart.destroy());
        this.charts.clear();
    }

    // Update chart data
    updateChart(canvasId, newData) {
        const chart = this.charts.get(canvasId);
        if (chart) {
            chart.data = newData;
            chart.update('active');
        }
    }

    // Update chart data smoothly
    updateChartDataset(canvasId, datasetIndex, newData) {
        const chart = this.charts.get(canvasId);
        if (chart && chart.data.datasets[datasetIndex]) {
            chart.data.datasets[datasetIndex].data = newData;
            chart.update('active');
        }
    }

    // Add data point to chart
    addDataPoint(canvasId, label, data) {
        const chart = this.charts.get(canvasId);
        if (chart) {
            chart.data.labels.push(label);
            chart.data.datasets.forEach((dataset, index) => {
                dataset.data.push(data[index] || 0);
            });
            chart.update('active');
        }
    }

    // Remove data point from chart
    removeDataPoint(canvasId) {
        const chart = this.charts.get(canvasId);
        if (chart) {
            chart.data.labels.pop();
            chart.data.datasets.forEach(dataset => {
                dataset.data.pop();
            });
            chart.update('active');
        }
    }

    // Resize all charts (useful for responsive layouts)
    resizeAllCharts() {
        this.charts.forEach(chart => chart.resize());
    }

    // Create animated progress ring
    createProgressRing(elementId, percentage, color = '#3B82F6', size = 120) {
        const element = document.getElementById(elementId);
        if (!element) return null;

        const radius = (size - 20) / 2;
        const circumference = 2 * Math.PI * radius;
        const strokeDasharray = `${circumference} ${circumference}`;
        const strokeDashoffset = circumference - (percentage / 100) * circumference;

        element.innerHTML = `
            <svg width="${size}" height="${size}" class="progress-ring">
                <defs>
                    <linearGradient id="gradient-${elementId}" x1="0%" y1="0%" x2="100%" y2="100%">
                        <stop offset="0%" style="stop-color:${color};stop-opacity:1" />
                        <stop offset="100%" style="stop-color:${color};stop-opacity:0.6" />
                    </linearGradient>
                </defs>
                <circle
                    class="progress-ring-bg"
                    stroke="#e5e7eb"
                    stroke-width="8"
                    fill="transparent"
                    r="${radius}"
                    cx="${size/2}"
                    cy="${size/2}"/>
                <circle
                    class="progress-ring-progress"
                    stroke="url(#gradient-${elementId})"
                    stroke-width="8"
                    fill="transparent"
                    r="${radius}"
                    cx="${size/2}"
                    cy="${size/2}"
                    style="stroke-dasharray: ${strokeDasharray}; stroke-dashoffset: ${strokeDashoffset}; transition: stroke-dashoffset 1s ease-in-out;"
                    transform="rotate(-90 ${size/2} ${size/2})"/>
                <text x="${size/2}" y="${size/2}" text-anchor="middle" dy="0.3em" 
                      class="text-2xl font-bold fill-current text-gray-800">
                    ${Math.round(percentage)}%
                </text>
            </svg>
        `;

        // Animate the ring
        const progressCircle = element.querySelector('.progress-ring-progress');
        if (progressCircle) {
            progressCircle.style.strokeDashoffset = circumference;
            
            setTimeout(() => {
                progressCircle.style.strokeDashoffset = strokeDashoffset;
            }, 100);
        }

        return element;
    }

    // Create health score gauge
    createHealthGauge(elementId, score, size = 200) {
        const element = document.getElementById(elementId);
        if (!element) return null;

        const angle = (score / 100) * 180 - 90; // Convert to gauge angle
        let color = '#10B981'; // Default green
        
        if (score < 30) color = '#EF4444'; // Red
        else if (score < 60) color = '#F59E0B'; // Yellow
        else if (score < 80) color = '#3B82F6'; // Blue

        element.innerHTML = `
            <div class="health-score-meter relative" style="width: ${size}px; height: ${size/2}px;">
                <div class="health-score-arc absolute inset-0 rounded-t-full border-8" 
                     style="border-color: #e5e7eb; border-bottom-color: transparent;"></div>
                <div class="health-score-arc absolute inset-0 rounded-t-full border-8" 
                     style="border-color: ${color}; border-bottom-color: transparent; 
                            clip-path: polygon(0 0, ${score}% 0, ${score}% 100%, 0 100%);"></div>
                <div class="health-score-needle absolute bottom-0 left-1/2 w-0.5 bg-gray-800 origin-bottom transition-transform duration-1000"
                     style="height: ${size/2 - 20}px; transform: translate(-50%, 0) rotate(${angle}deg);"></div>
                <div class="absolute bottom-2 left-1/2 transform -translate-x-1/2">
                    <div class="text-3xl font-bold text-gray-800">${Math.round(score)}</div>
                    <div class="text-sm text-gray-600 text-center">Health Score</div>
                </div>
            </div>
        `;

        return element;
    }

    // Get chart instance
    getChart(canvasId) {
        return this.charts.get(canvasId);
    }

    // Check if chart exists
    hasChart(canvasId) {
        return this.charts.has(canvasId);
    }

    // Get all chart IDs
    getChartIds() {
        return Array.from(this.charts.keys());
    }

    // Download chart as image
    downloadChart(canvasId, filename = 'chart.png') {
        const chart = this.charts.get(canvasId);
        if (chart) {
            const url = chart.toBase64Image();
            const link = document.createElement('a');
            link.download = filename;
            link.href = url;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        }
    }
}

// Global utility functions for chart updates
function updateTrendChart(data) {
    const chartRenderer = window.chartRenderer || new ChartRenderer();
    return chartRenderer.createRiskTrendChart('risk-trend-chart', data);
}

function updateRiskFactorsChart(data) {
    const chartRenderer = window.chartRenderer || new ChartRenderer();
    return chartRenderer.createRiskFactorsRadarChart('risk-factors-chart', data);
}

function updateTaskCompletionChart(data) {
    const chartRenderer = window.chartRenderer || new ChartRenderer();
    return chartRenderer.createTaskCompletionChart('task-completion-chart', data);
}

function updateWeeklyProgressChart(data) {
    const chartRenderer = window.chartRenderer || new ChartRenderer();
    return chartRenderer.createWeeklyProgressChart('weekly-progress-chart', data);
}

function updateModelComparisonChart(data) {
    const chartRenderer = window.chartRenderer || new ChartRenderer();
    return chartRenderer.createModelComparisonChart('model-comparison-chart', data);
}

function createHealthScoreGauge(elementId, score) {
    const chartRenderer = window.chartRenderer || new ChartRenderer();
    return chartRenderer.createHealthGauge(elementId, score);
}

function createProgressRing(elementId, percentage, color) {
    const chartRenderer = window.chartRenderer || new ChartRenderer();
    return chartRenderer.createProgressRing(elementId, percentage, color);
}

// Initialize chart renderer globally
window.ChartRenderer = ChartRenderer;
window.chartRenderer = new ChartRenderer();

// Chart data processing utilities
const ChartUtils = {
    // Process prediction data for charts
    processPredictionData: function(predictions) {
        const models = Object.keys(predictions);
        const probabilities = models.map(model => predictions[model].probability * 100);}
    }