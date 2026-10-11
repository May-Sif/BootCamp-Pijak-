# modelling_tuning.py
"""
Model training dengan hyperparameter tuning dan manual logging
"""

import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)

import dagshub
# Setup DagsHub tracking
dagshub.init(
    repo_name='Eksperimen_SML_Maya',
    repo_owner='mayasifa',  # Ganti!
    mlflow=True
)

# Load data
print("Loading data...")
df = pd.read_csv("heart_preprocessing.csv")
X = df.drop("target", axis=1)
y = df["target"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Hyperparameter grid
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, 10, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

# GridSearchCV
grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1
)

# MLflow tracking
with mlflow.start_run(run_name="Tuned_RandomForest"):
    print("\nRunning GridSearchCV...")
    
    # Fit grid search
    grid.fit(X_train, y_train)
    
    # Best model and parameters
    best_model = grid.best_estimator_
    best_params = grid.best_params_
    
    print(f"\nBest parameters: {best_params}")
    
    # Predictions
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"\nTest Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    # Log parameters
    for param_name, param_value in best_params.items():
        mlflow.log_param(param_name, param_value)
    
    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("best_cv_score", grid.best_score_)
    
    # --- ARTIFACT 1: Classification Report ---
    report = classification_report(y_test, y_pred)
    with open("classification_report.txt", "w") as f:
        f.write(report)
    mlflow.log_artifact("classification_report.txt")
    
    # --- ARTIFACT 2: Confusion Matrix Plot ---
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    plt.tight_layout()
    mlflow.log_figure(fig, "confusion_matrix.png")
    plt.close()
    
    # --- ARTIFACT 3: ROC Curve (ADVANCE - tambahan) ---
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax2.legend(loc="lower right")
    plt.tight_layout()
    mlflow.log_figure(fig2, "roc_curve.png")
    plt.close()
    
    # --- ARTIFACT 4: Feature Importance ---
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.barplot(data=feature_importance.head(10), x='importance', y='feature', ax=ax3)
    ax3.set_title('Top 10 Feature Importance')
    plt.tight_layout()
    mlflow.log_figure(fig3, "feature_importance.png")
    plt.close()
    
    # Log feature importance as CSV
    feature_importance.to_csv("feature_importance.csv", index=False)
    mlflow.log_artifact("feature_importance.csv")
    
    # Save model
    mlflow.sklearn.log_model(best_model, artifact_path="model")
    
    print("\nAll artifacts logged successfully!")
    print(f"Check MLflow UI for: classification_report.txt, confusion_matrix.png, roc_curve.png, feature_importance.png")

print("\nTraining completed!")