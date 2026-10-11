# inference.py - Model serving script tanpa MLflow Registry

import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
import pickle
import os
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

def load_or_train_model():
    """Load model from local file or train new one"""
    
    model_path = "model.pkl"
    
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    
    print("No saved model found. Training new model...")
    
    dataset_path = "heart_preprocessing.csv"
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}")
        print("Please copy heart_preprocessing.csv to this folder")
        return None
    
    df = pd.read_csv(dataset_path)
    X = df.drop("target", axis=1)
    y = df["target"]
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"New model trained and saved as {model_path}")
    
    return model

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "service": "Heart Disease Prediction API"
    }), 200

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint"""
    try:
        data = request.get_json()
        
        if 'features' not in data:
            return jsonify({
                "error": "Missing 'features' field",
                "status": "failed"
            }), 400
        
        features = data['features']
        
        feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
                         'restecg', 'thalach', 'exang', 'oldpeak', 
                         'slope', 'ca', 'thal']
        
        if len(features) != len(feature_names):
            return jsonify({
                "error": f"Expected {len(feature_names)} features, got {len(features)}",
                "expected_features": feature_names,
                "status": "failed"
            }), 400
        
        features_df = pd.DataFrame([features], columns=feature_names)
        
        prediction = model.predict(features_df)[0]
        
        probability = None
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(features_df)[0]
            probability = {
                "no_heart_disease": float(proba[0]),
                "heart_disease": float(proba[1])
            }
        
        return jsonify({
            "prediction": int(prediction),
            "label": "Heart Disease" if prediction == 1 else "No Heart Disease",
            "probability": probability,
            "status": "success"
        }), 200
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "failed"
        }), 400

@app.route('/model_info', methods=['GET'])
def model_info():
    """Get model information"""
    return jsonify({
        "model_type": type(model).__name__,
        "features_count": 13,
        "features": ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
                     'restecg', 'thalach', 'exang', 'oldpeak', 
                     'slope', 'ca', 'thal']
    }), 200

if __name__ == "__main__":
    print("\n" + "="*50)
    print("Starting Heart Disease Prediction Server")
    print("="*50)
    
    model = load_or_train_model()
    
    if model is None:
        print("Failed to load/train model. Exiting...")
        exit(1)
    
    print(f"Model loaded: {type(model).__name__}")
    print(f"API Endpoint: http://localhost:5000")
    print(f"Health check: http://localhost:5000/health")
    print(f"Prediction: POST http://localhost:5000/predict")
    print(f"Model info: GET http://localhost:5000/model_info")
    print("="*50 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True)