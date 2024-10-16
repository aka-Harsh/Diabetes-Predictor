document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('prediction-form');
    const result = document.getElementById('result');
    const predictionSpan = document.getElementById('prediction');
    const topFactorsList = document.getElementById('top-factors');
    const toggleSwitch = document.querySelector('.theme-switch input[type="checkbox"]');

    form.addEventListener('submit', function(e) {
        e.preventDefault();

        const formData = {
            gender: document.getElementById('gender').value,
            age: parseFloat(document.getElementById('age').value),
            hypertension: parseInt(document.getElementById('hypertension').value),
            heart_disease: parseInt(document.getElementById('heart_disease').value),
            smoking_history: document.getElementById('smoking_history').value,
            bmi: parseFloat(document.getElementById('bmi').value),
            HbA1c_level: parseFloat(document.getElementById('HbA1c_level').value),
            blood_glucose_level: parseFloat(document.getElementById('blood_glucose_level').value)
        };

        fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(formData),
        })
        .then(response => response.json())
        .then(data => {
            predictionSpan.textContent = data.prediction;
            topFactorsList.innerHTML = '';
            data.top_factors.forEach(factor => {
                const li = document.createElement('li');
                li.textContent = factor;
                topFactorsList.appendChild(li);
            });
            result.classList.remove('hidden');
            result.classList.add('fade-in');
        })
        .catch((error) => {
            console.error('Error:', error);
        });
    });

    // Dark mode toggle
    function switchTheme(e) {
        if (e.target.checked) {
            document.body.classList.add('dark-mode');
            localStorage.setItem('theme', 'dark');
        } else {
            document.body.classList.remove('dark-mode');
            localStorage.setItem('theme', 'light');
        }    
    }

    toggleSwitch.addEventListener('change', switchTheme, false);

    // Check for saved user preference, if any, on load of the website
    const currentTheme = localStorage.getItem('theme');
    if (currentTheme) {
        document.body.classList[currentTheme === 'dark' ? 'add' : 'remove']('dark-mode');
        toggleSwitch.checked = currentTheme === 'dark';
    }
});