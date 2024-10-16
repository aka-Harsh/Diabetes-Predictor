import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from flask import Flask, render_template, request, jsonify
import joblib
import os

# Load the data
data = pd.read_csv('diabetes_data.csv')

# Separate features and target
X = data.drop('diabetes', axis=1)
y = data['diabetes']

# Define categorical and numerical columns
categorical_features = ['gender', 'smoking_history']
numerical_features = ['age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level', 'blood_glucose_level']

# Create preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
    ])

# Create pipeline
model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the model
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'diabetes_model.joblib')

# Set up Flask application
app = Flask(__name__, template_folder=os.path.abspath('templates'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_data = pd.DataFrame([data])
    
    prediction = model.predict_proba(input_data)[0][1]
    
    feature_names = model.named_steps['preprocessor'].get_feature_names_out()
    feature_importance = model.named_steps['classifier'].feature_importances_
    
    sorted_idx = feature_importance.argsort()
    top_features = [(feature_names[i], feature_importance[i]) for i in sorted_idx[-3:]]
    top_features.reverse()
    
    # Format feature names
    formatted_features = []
    for feature, importance in top_features:
        if feature.startswith('num__'):
            feature = feature.replace('num__', '')
        elif feature.startswith('cat__'):
            feature = feature.replace('cat__', '')
        formatted_features.append(f"{feature}: {importance:.2f}")
    
    return jsonify({
        'prediction': f"{prediction:.2%}",
        'top_factors': formatted_features
    })

if __name__ == '__main__':
    app.run(debug=True)