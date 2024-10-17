#src/train.py

import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import json

def train_evaluate_model(model, X_train, y_train, X_test, y_test, model_name, sample_weights=None):
    if sample_weights is not None:
        model.fit(X_train, y_train, sample_weight=sample_weights)
    else:
        model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Sem Vítimas', 'Com Vítimas Feridas', 'Com Vítimas Fatais'], output_dict=True)

    print(f"Modelo: {model_name}")
    print("Accuracy:", accuracy)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Sem Vítimas', 'Com Vítimas Feridas', 'Com Vítimas Fatais']))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Sem Vítimas', 'Com Vítimas Feridas', 'Com Vítimas Fatais'],
                yticklabels=['Sem Vítimas', 'Com Vítimas Feridas', 'Com Vítimas Fatais'])
    plt.xlabel('Predito')
    plt.ylabel('Verdadeiro')
    plt.title(f'Matriz de Confusão - {model_name}')
    plt.savefig(f"confusion_matrix_{model_name}.png")
    plt.close()

    return accuracy, report

def apply_sampling(sampling_technique, X_train, y_train):
    if sampling_technique == 'SMOTE':
        sampler = SMOTE(random_state=42)
    elif sampling_technique == 'SMOTEENN':
        sampler = SMOTEENN(random_state=42)
    else:
        return X_train, y_train

    X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
    return X_resampled, y_resampled

def perform_randomized_search(model, param_distributions, X, y, sample_weights=None, scoring='f1_macro'):
    randomized_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=50,
        cv=3,
        scoring=scoring,
        n_jobs=-1,
        random_state=42
    )
    if sample_weights is not None:
        randomized_search.fit(X, y, sample_weight=sample_weights)
    else:
        randomized_search.fit(X, y)
    best_model = randomized_search.best_estimator_
    print(f"Melhores hiperparâmetros para {model.__class__.__name__}:")
    print(randomized_search.best_params_)
    return best_model



def train_model():
    mlruns_path = os.path.abspath("./mlruns")
    mlflow.set_tracking_uri(f"file://{mlruns_path}")  # Use o caminho absoluto

    X_train = pd.read_csv('../data/processed/train.csv')
    X_test = pd.read_csv('../data/processed/test.csv')
    y_train = pd.read_csv('../data/processed/train_labels.csv').values.ravel()
    y_test = pd.read_csv('../data/processed/test_labels.csv').values.ravel()

    class_weights = compute_class_weight('balanced', classes=np.array([0, 1, 2]), y=y_train)
    class_weights = dict(zip([0, 1, 2], class_weights))
    sample_weights = compute_sample_weight('balanced', y=y_train)

    models = {
        'Random Forest': RandomForestClassifier(class_weight=class_weights, random_state=42),
        'XGBoost': xgb.XGBClassifier(eval_metric='mlogloss', random_state=42),
        'Logistic Regression': LogisticRegression(class_weight=class_weights, max_iter=5000, random_state=42)
    }

    mlflow.set_experiment("Acidentes de Trânsito")

    results = {}

    for model_name, model in models.items():
        with mlflow.start_run(run_name=f"{model_name} com Class Weights"):
            if model_name == 'XGBoost':
                accuracy, report = train_evaluate_model(
                    model, X_train, y_train, X_test, y_test, model_name + " com Class Weights", sample_weights=sample_weights)
            else:
                accuracy, report = train_evaluate_model(
                    model, X_train, y_train, X_test, y_test, model_name + " com Class Weights")
            results[(model_name, 'Class Weights')] = report

            mlflow.log_param("sampling", "None")
            mlflow.log_param("class_weight", "Balanced")
            mlflow.log_param("model_name", model_name)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("f1_score_fatal", report['Com Vítimas Fatais']['f1-score'])
            mlflow.log_metric("recall_fatal", report['Com Vítimas Fatais']['recall'])
            mlflow.log_metric("precision_fatal", report['Com Vítimas Fatais']['precision'])
            mlflow.log_artifact(f"confusion_matrix_{model_name} com Class Weights.png")

            if model_name == 'XGBoost':
                mlflow.xgboost.log_model(model, artifact_path="model")
            else:
                mlflow.sklearn.log_model(model, artifact_path="model")

    X_train_smote, y_train_smote = apply_sampling('SMOTE', X_train, y_train)
    class_weights_smote = compute_class_weight('balanced', classes=np.unique(y_train_smote), y=y_train_smote)
    class_weights_smote = dict(zip(np.unique(y_train_smote), class_weights_smote))
    sample_weights_smote = compute_sample_weight('balanced', y=y_train_smote)

    for model_name, model in models.items():
        with mlflow.start_run(run_name=f"{model_name} com SMOTE e Class Weights"):
            if model_name == 'XGBoost':
                model = xgb.XGBClassifier(eval_metric='mlogloss', random_state=42)
                accuracy, report = train_evaluate_model(
                    model, X_train_smote, y_train_smote, X_test, y_test, model_name + " com SMOTE e Class Weights", sample_weights=sample_weights_smote)
            else:
                model.set_params(class_weight=class_weights_smote)
                accuracy, report = train_evaluate_model(
                    model, X_train_smote, y_train_smote, X_test, y_test, model_name + " com SMOTE e Class Weights")
            results[(model_name, 'SMOTE + Class Weights')] = report

            mlflow.log_param("sampling", "SMOTE")
            mlflow.log_param("class_weight", "Balanced")
            mlflow.log_param("model_name", model_name)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("f1_score_fatal", report['Com Vítimas Fatais']['f1-score'])
            mlflow.log_metric("recall_fatal", report['Com Vítimas Fatais']['recall'])
            mlflow.log_metric("precision_fatal", report['Com Vítimas Fatais']['precision'])
            mlflow.log_artifact(f"confusion_matrix_{model_name} com SMOTE e Class Weights.png")

            if model_name == 'XGBoost':
                mlflow.xgboost.log_model(model, artifact_path="model")
            else:
                mlflow.sklearn.log_model(model, artifact_path="model")

    X_train_smoteenn, y_train_smoteenn = apply_sampling('SMOTEENN', X_train, y_train)
    class_weights_smoteenn = compute_class_weight('balanced', classes=np.unique(y_train_smoteenn), y=y_train_smoteenn)
    class_weights_smoteenn = dict(zip(np.unique(y_train_smoteenn), class_weights_smoteenn))
    sample_weights_smoteenn = compute_sample_weight('balanced', y=y_train_smoteenn)

    param_distributions_rf = {
        'n_estimators': [100, 200, 300, 400, 500],
        'max_depth': [None, 10, 20, 30, 40, 50],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }

    param_distributions_xgb = {
        'n_estimators': [100, 200, 300, 400, 500],
        'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
        'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0]
    }

    for model_name, model in models.items():
        with mlflow.start_run(run_name=f"{model_name} com SMOTEENN e Class Weights"):
            if model_name == 'XGBoost':
                model = xgb.XGBClassifier(eval_metric='mlogloss', random_state=42)
                model = perform_randomized_search(model, param_distributions_xgb, X_train_smoteenn, y_train_smoteenn, sample_weights=sample_weights_smoteenn)
                accuracy, report = train_evaluate_model(
                    model, X_train_smoteenn, y_train_smoteenn, X_test, y_test, model_name + " com SMOTEENN e Class Weights", sample_weights=sample_weights_smoteenn)
            elif model_name == 'Random Forest':
                model = perform_randomized_search(model, param_distributions_rf, X_train_smoteenn, y_train_smoteenn, sample_weights=sample_weights_smoteenn)
                accuracy, report = train_evaluate_model(
                    model, X_train_smoteenn, y_train_smoteenn, X_test, y_test, model_name + " com SMOTEENN e Class Weights")
            results[(model_name, 'SMOTEENN + Class Weights')] = report

            mlflow.log_param("sampling", "SMOTEENN")
            mlflow.log_param("class_weight", "Balanced")
            mlflow.log_param("model_name", model_name)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("f1_score_fatal", report['Com Vítimas Fatais']['f1-score'])
            mlflow.log_metric("recall_fatal", report['Com Vítimas Fatais']['recall'])
            mlflow.log_metric("precision_fatal", report['Com Vítimas Fatais']['precision'])
            mlflow.log_artifact(f"confusion_matrix_{model_name} com SMOTEENN e Class Weights.png")

            if model_name == 'XGBoost':
                mlflow.xgboost.log_model(model, artifact_path="model")
            else:
                mlflow.sklearn.log_model(model, artifact_path="model")


if __name__ == "__main__":
    train_model()

