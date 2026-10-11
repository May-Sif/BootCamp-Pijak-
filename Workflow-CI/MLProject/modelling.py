"""
Model training untuk CI/CD pipeline
"""

import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import pickle
import argparse
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=10)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()
    
    print("="*50)
    print("Training Heart Disease Model")
    print(f"Parameters: n_estimators={args.n_estimators}, max_depth={args.max_depth}")
    print("="*50)
    
    # Load data
    print("Loading data...")
    df = pd.read_csv("heart_preprocessing.csv")
    X = df.drop("target", axis=1)
    y = df["target"]
    print(f"   Shape: {X.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, 
        random_state=args.random_state, stratify=y
    )
    print(f"   Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Train model with MLflow tracking
    with mlflow.start_run():
        # Log parameters
        mlflow.log_params({
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "test_size": args.test_size,
            "random_state": args.random_state
        })
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=args.random_state
        )
        model.fit(X_train, y_train)
        print("Model trained")
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Log metrics
        mlflow.log_metrics({
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        })
        
        print(f"\nResults:")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall: {recall:.4f}")
        print(f"   F1-Score: {f1:.4f}")
        
        # Save model as pickle file (for inference)
        with open("model.pkl", "wb") as f:
            pickle.dump(model, f)
        print("Model saved as model.pkl")
        
        # Save metrics to txt file
        with open("metrics.txt", "w") as f:
            f.write(f"Accuracy: {accuracy}\n")
            f.write(f"Precision: {precision}\n")
            f.write(f"Recall: {recall}\n")
            f.write(f"F1-Score: {f1}\n")
            f.write(f"n_estimators: {args.n_estimators}\n")
            f.write(f"max_depth: {args.max_depth}\n")
        print("Metrics saved as metrics.txt")
        
        # Save model with MLflow
        mlflow.sklearn.log_model(model, "model")
        print("Model logged to MLflow")
    
    print("="*50)
    print("Training completed successfully!")
    print("="*50)

if __name__ == "__main__":
    main()