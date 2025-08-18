// Form Wizard for Multi-Step Prediction Form
class FormWizard {
    constructor(formId) {
        this.form = document.getElementById(formId);
        this.steps = this.form.querySelectorAll('.form-step');
        this.currentStep = 1;
        this.totalSteps = this.steps.length;
        
        this.prevBtn = document.getElementById('prev-btn');
        this.nextBtn = document.getElementById('next-btn');
        this.submitBtn = document.getElementById('submit-btn');
        this.progressBar = document.getElementById('progress-bar');
        
        this.init();
    }
    
    init() {
        this.updateUI();
        this.attachEventListeners();
        this.setupValidation();
    }
    
    attachEventListeners() {
        if (this.prevBtn) {
            this.prevBtn.addEventListener('click', () => this.previousStep());
        }
        
        if (this.nextBtn) {
            this.nextBtn.addEventListener('click', () => this.nextStep());
        }
        
        // Enter key navigation
        this.form.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.target.matches('textarea')) {
                e.preventDefault();
                if (this.currentStep < this.totalSteps) {
                    this.nextStep();
                }
            }
        });
        
        // Auto-advance on radio button selection
        this.form.addEventListener('change', (e) => {
            if (e.target.type === 'radio') {
                // Small delay to let user see their selection
                setTimeout(() => {
                    if (this.validateCurrentStep() && this.currentStep < this.totalSteps) {
                        this.nextStep();
                    }
                }, 500);
            }
        });
    }
    
    setupValidation() {
        // Real-time validation
        const inputs = this.form.querySelectorAll('input, select, textarea');
        inputs.forEach(input => {
            input.addEventListener('blur', () => this.validateField(input));
            input.addEventListener('input', () => this.clearFieldError(input));
        });
    }
    
    nextStep() {
        if (!this.validateCurrentStep()) {
            this.showValidationErrors();
            return;
        }
        
        if (this.currentStep < this.totalSteps) {
            this.currentStep++;
            this.updateUI();
            this.animateStepTransition('forward');
        }
    }
    
    previousStep() {
        if (this.currentStep > 1) {
            this.currentStep--;
            this.updateUI();
            this.animateStepTransition('backward');
        }
    }
    
    updateUI() {
        // Update step visibility
        this.steps.forEach((step, index) => {
            const stepNumber = index + 1;
            if (stepNumber === this.currentStep) {
                step.classList.add('active');
            } else {
                step.classList.remove('active');
            }
        });
        
        // Update step indicators
        const indicators = document.querySelectorAll('.step-indicator');
        indicators.forEach((indicator, index) => {
            const stepNumber = index + 1;
            indicator.classList.remove('active', 'completed');
            
            if (stepNumber < this.currentStep) {
                indicator.classList.add('completed');
                indicator.innerHTML = '✓';
            } else if (stepNumber === this.currentStep) {
                indicator.classList.add('active');
                indicator.innerHTML = stepNumber;
            } else {
                indicator.innerHTML = stepNumber;
            }
        });
        
        // Update progress bar
        const progress = (this.currentStep / this.totalSteps) * 100;
        if (this.progressBar) {
            this.progressBar.style.width = `${progress}%`;
        }
        
        // Update navigation buttons
        if (this.prevBtn) {
            if (this.currentStep === 1) {
                this.prevBtn.classList.add('hidden');
            } else {
                this.prevBtn.classList.remove('hidden');
            }
        }
        
        if (this.nextBtn && this.submitBtn) {
            if (this.currentStep === this.totalSteps) {
                this.nextBtn.classList.add('hidden');
                this.submitBtn.classList.remove('hidden');
            } else {
                this.nextBtn.classList.remove('hidden');
                this.submitBtn.classList.add('hidden');
            }
        }
        
        // Focus on first input of current step
        this.focusFirstInput();
    }
    
    animateStepTransition(direction) {
        const currentStepEl = this.steps[this.currentStep - 1];
        
        // Add animation classes
        currentStepEl.style.opacity = '0';
        currentStepEl.style.transform = direction === 'forward' ? 'translateX(-20px)' : 'translateX(20px)';
        
        setTimeout(() => {
            currentStepEl.style.opacity = '1';
            currentStepEl.style.transform = 'translateX(0)';
        }, 150);
    }
    
    focusFirstInput() {
        const currentStepEl = this.steps[this.currentStep - 1];
        const firstInput = currentStepEl.querySelector('input:not([type="hidden"]), select, textarea');
        
        if (firstInput && !firstInput.disabled) {
            setTimeout(() => {
                firstInput.focus();
            }, 200);
        }
    }
    
    validateCurrentStep() {
        const currentStepEl = this.steps[this.currentStep - 1];
        const requiredFields = currentStepEl.querySelectorAll('[required]');
        let isValid = true;
        
        requiredFields.forEach(field => {
            if (!this.validateField(field)) {
                isValid = false;
            }
        });
        
        // Step-specific validation
        if (this.currentStep === 1) {
            isValid = this.validateStep1() && isValid;
        } else if (this.currentStep === 2) {
            isValid = this.validateStep2() && isValid;
        } else if (this.currentStep === 3) {
            isValid = this.validateStep3() && isValid;
        } else if (this.currentStep === 4) {
            isValid = this.validateStep4() && isValid;
        }
        
        return isValid;
    }
    
    validateField(field) {
        const value = field.value.trim();
        let isValid = true;
        let errorMessage = '';
        
        // Clear previous errors
        this.clearFieldError(field);
        
        // Required field validation
        if (field.hasAttribute('required') && !value) {
            isValid = false;
            errorMessage = 'This field is required';
        }
        
        // Type-specific validation
        if (isValid && value) {
            switch (field.type) {
                case 'number':
                    const num = parseFloat(value);
                    const min = parseFloat(field.min);
                    const max = parseFloat(field.max);
                    
                    if (isNaN(num)) {
                        isValid = false;
                        errorMessage = 'Please enter a valid number';
                    } else if (!isNaN(min) && num < min) {
                        isValid = false;
                        errorMessage = `Value must be at least ${min}`;
                    } else if (!isNaN(max) && num > max) {
                        isValid = false;
                        errorMessage = `Value must be no more than ${max}`;
                    }
                    break;
                    
                case 'email':
                    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
                    if (!emailRegex.test(value)) {
                        isValid = false;
                        errorMessage = 'Please enter a valid email address';
                    }
                    break;
            }
        }
        
        // Field-specific validation
        if (isValid && value) {
            switch (field.id) {
                case 'bmi':
                    const bmi = parseFloat(value);
                    if (bmi < 10 || bmi > 60) {
                        isValid = false;
                        errorMessage = 'BMI seems unrealistic. Please check your entry.';
                    }
                    break;
                    
                case 'age':
                    const age = parseInt(value);
                    if (age < 18 || age > 120) {
                        isValid = false;
                        errorMessage = 'Please enter a valid age between 18 and 120';
                    }
                    break;
                    
                case 'hba1c_level':
                    const hba1c = parseFloat(value);
                    if (hba1c > 0 && (hba1c < 3 || hba1c > 20)) {
                        isValid = false;
                        errorMessage = 'HbA1c value seems unrealistic. Please check your entry.';
                    }
                    break;
                    
                case 'blood_glucose_level':
                    const glucose = parseFloat(value);
                    if (glucose > 0 && (glucose < 50 || glucose > 500)) {
                        isValid = false;
                        errorMessage = 'Blood glucose value seems unrealistic. Please check your entry.';
                    }
                    break;
            }
        }
        
        // Show error if validation failed
        if (!isValid) {
            this.showFieldError(field, errorMessage);
        }
        
        return isValid;
    }
    
    validateStep1() {
        // Validate BMI calculation consistency
        const height = document.getElementById('height');
        const weight = document.getElementById('weight');
        const bmi = document.getElementById('bmi');
        
        if (height.value && weight.value && bmi.value) {
            const calculatedBMI = parseFloat(weight.value) / Math.pow(parseFloat(height.value) / 100, 2);
            const enteredBMI = parseFloat(bmi.value);
            
            if (Math.abs(calculatedBMI - enteredBMI) > 1) {
                this.showFieldError(bmi, 'BMI doesn\'t match height and weight. Please recalculate.');
                return false;
            }
        }
        
        return true;
    }
    
    validateStep2() {
        // Check if at least some medical history is provided
        const hypertension = document.querySelector('input[name="hypertension"]:checked');
        const heartDisease = document.querySelector('input[name="heart_disease"]:checked');
        
        if (!hypertension && !heartDisease) {
            this.showStepError(2, 'Please provide information about your medical history');
            return false;
        }
        
        return true;
    }
    
    validateStep3() {
        // Validate lab values consistency
        const hba1c = document.getElementById('hba1c_level');
        const glucose = document.getElementById('blood_glucose_level');
        
        if (hba1c.value && glucose.value) {
            const hba1cVal = parseFloat(hba1c.value);
            const glucoseVal = parseFloat(glucose.value);
            
            // Rough correlation check
            const estimatedHbA1c = (glucoseVal + 46.7) / 28.7;
            
            if (Math.abs(hba1cVal - estimatedHbA1c) > 2.5) {
                this.showStepWarning(3, 'HbA1c and blood glucose values may not be consistent. Please verify your entries.');
            }
        }
        
        return true;
    }
    
    validateStep4() {
        // Check for reasonable lifestyle combinations
        const activity = document.getElementById('physical_activity');
        const sleep = document.getElementById('sleep_hours');
        
        if (activity.value === 'extremely_active' && sleep.value && parseFloat(sleep.value) < 6) {
            this.showStepWarning(4, 'Very high activity with low sleep may not be sustainable for health.');
        }
        
        return true;
    }
    
    showFieldError(field, message) {
        // Remove existing error
        this.clearFieldError(field);
        
        // Add error styling
        field.classList.add('border-red-500', 'bg-red-50');
        
        // Create error message element
        const errorEl = document.createElement('div');
        errorEl.className = 'mt-1 text-sm text-red-600';
        errorEl.textContent = message;
        errorEl.id = `error-${field.id}`;
        
        // Insert error message after field
        field.parentNode.appendChild(errorEl);
    }
    
    clearFieldError(field) {
        field.classList.remove('border-red-500', 'bg-red-50');
        const existingError = document.getElementById(`error-${field.id}`);
        if (existingError) {
            existingError.remove();
        }
    }
    
    showStepError(stepNumber, message) {
        this.showStepMessage(stepNumber, message, 'error');
    }
    
    showStepWarning(stepNumber, message) {
        this.showStepMessage(stepNumber, message, 'warning');
    }
    
    showStepMessage(stepNumber, message, type) {
        const stepEl = this.steps[stepNumber - 1];
        const existingMessage = stepEl.querySelector('.step-message');
        
        if (existingMessage) {
            existingMessage.remove();
        }
        
        const messageEl = document.createElement('div');
        const bgColor = type === 'error' ? 'bg-red-50 border-red-200 text-red-700' : 'bg-yellow-50 border-yellow-200 text-yellow-700';
        
        messageEl.className = `step-message mt-4 p-3 rounded-lg border ${bgColor}`;
        messageEl.innerHTML = `
            <div class="flex">
                <span class="mr-2">${type === 'error' ? '❌' : '⚠️'}</span>
                <span class="text-sm">${message}</span>
            </div>
        `;
        
        stepEl.insertBefore(messageEl, stepEl.firstChild.nextSibling);
        
        // Auto-remove after 5 seconds for warnings
        if (type === 'warning') {
            setTimeout(() => {
                if (messageEl.parentNode) {
                    messageEl.remove();
                }
            }, 5000);
        }
    }
    
    showValidationErrors() {
        // Show toast notification
        if (typeof showToast === 'function') {
            showToast('Please fix the errors before continuing', 'error');
        }
        
        // Focus on first error field
        const currentStepEl = this.steps[this.currentStep - 1];
        const firstErrorField = currentStepEl.querySelector('.border-red-500');
        
        if (firstErrorField) {
            firstErrorField.focus();
            firstErrorField.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }
    }
    
    // Public method to get form data
    getFormData() {
        const formData = new FormData(this.form);
        const data = {};
        
        for (let [key, value] of formData.entries()) {
            data[key] = value;
        }
        
        return data;
    }
    
    // Public method to populate form
    populateForm(data) {
        Object.entries(data).forEach(([key, value]) => {
            const field = this.form.querySelector(`[name="${key}"]`);
            if (field) {
                if (field.type === 'radio') {
                    const radio = this.form.querySelector(`[name="${key}"][value="${value}"]`);
                    if (radio) {
                        radio.checked = true;
                    }
                } else {
                    field.value = value;
                }
            }
        });
        
        // Trigger change events to update calculated fields
        this.form.dispatchEvent(new Event('change', { bubbles: true }));
    }
    
    // Public method to go to specific step
    goToStep(stepNumber) {
        if (stepNumber >= 1 && stepNumber <= this.totalSteps) {
            this.currentStep = stepNumber;
            this.updateUI();
        }
    }
    
    // Public method to reset form
    reset() {
        this.form.reset();
        this.currentStep = 1;
        this.updateUI();
        
        // Clear all errors
        const errorElements = this.form.querySelectorAll('[id^="error-"], .step-message');
        errorElements.forEach(el => el.remove());
        
        // Clear field error styling
        const errorFields = this.form.querySelectorAll('.border-red-500, .bg-red-50');
        errorFields.forEach(field => {
            field.classList.remove('border-red-500', 'bg-red-50');
        });
    }
}

// Export for global access
window.FormWizard = FormWizard;