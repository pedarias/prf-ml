# train_SMOTE_CW.py

import os
import mlflow
from train_utils import (
    train_evaluate_model, apply_sampling, load_data, load_preprocessor, log_mlflow
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier


def train_model():
    # Configurar o rastreamento do MLflow
    mlflow.set_tracking_uri("file://" + os.path.abspath("./mlruns"))
    mlflow.set_experiment("Acidentes de Trânsito - SMOTE + Class Weights")

    # Carregar os dados
    X_train_raw, X_test_raw, y_train, y_test = load_data()
    preprocessor = load_preprocessor()

    # Aplicar o SMOTE
    X_train_resampled, y_train_resampled = apply_sampling('SMOTE', X_train_raw, y_train, preprocessor)

    # Definir os pesos das classes
    class_weights = {0: 1, 1: 1, 2: 5}

    # Definir os modelos
    models = {
        'Random Forest': RandomForestClassifier(class_weight=class_weights, random_state=42, n_jobs=-1),
        'Logistic Regression': LogisticRegression(class_weight=class_weights, max_iter=1000, random_state=42, n_jobs=-1),
        'XGBoost': xgb.XGBClassifier(eval_metric='mlogloss', scale_pos_weight=class_weights[2], random_state=42, n_jobs=-1),
        'LightGBM': lgb.LGBMClassifier(class_weight=class_weights, random_state=42, n_jobs=-1, force_row_wise=True),
        'CatBoost': CatBoostClassifier(class_weights=[1, 1, 5], random_state=42, verbose=0, thread_count=-1)
    }

    for model_name, model in models.items():
        print(f"\nTreinando o modelo: {model_name}")

        # Treinar e avaliar o modelo
        accuracy, report, cm_filename = train_evaluate_model(
            model, X_train_resampled, y_train_resampled, X_test_raw, y_test,
            model_name, preprocessor, data_preprocessed=True
        )

        # Preparar parâmetros e métricas para o MLflow
        params = {
            "sampling": "SMOTE",
            "class_weight": class_weights,
            "model_name": model_name
        }
        metrics = {
            "accuracy": accuracy,
            "f1_score_fatal": report['Com Vítimas Fatais']['f1-score'],
            "recall_fatal": report['Com Vítimas Fatais']['recall'],
            "precision_fatal": report['Com Vítimas Fatais']['precision']
        }
        artifacts = {
            "confusion_matrix": cm_filename
        }

        # Registrar no MLflow
        log_mlflow(
            model_name,
            params,
            metrics,
            artifacts,
            model,
            preprocessor,
            run_name=f"{model_name} com SMOTE e Class Weights"
        )

if __name__ == "__main__":
    train_model()
