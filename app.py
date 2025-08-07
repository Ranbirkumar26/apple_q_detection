from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load model and scaler
model = joblib.load(os.path.join('models', 'apple_quality_model.pkl'))
scaler = joblib.load(os.path.join('models', 'scaler.pkl'))

# Feature order must match model training
numerical_features = ['Size', 'Weight', 'Sweetness', 'Crunchiness', 'Juiciness', 'Ripeness', 'Acidity']

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    confidence = None

    if request.method == 'POST':
        try:
            # Get and process input
            features = [float(request.form[feature]) for feature in numerical_features]
            X = np.array([features])
            X_scaled = scaler.transform(X)
            label = model.predict(X_scaled)[0]
            proba = model.predict_proba(X_scaled)[0]
            class_map = {0: "good", 1: "bad"}
            prediction = class_map[label]
            confidence = f"{proba[label]*100:.2f}%"
        except Exception as e:
            prediction = f"Error: {e}"

    return render_template('index.html', prediction=prediction, confidence=confidence, feature_names=numerical_features)

if __name__ == "__main__":
    app.run(debug=True)
