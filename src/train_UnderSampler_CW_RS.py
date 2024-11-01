# train_UnderSampler_CW_RS.py

import os
import mlflow
from train_utils import (
    train_evaluate_model, apply_sampling, load_data, load_preprocessor, log_mlflow,
    perform_randomized_search, perform_randomized_search_lgb
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

def train_model():
    # Configurar o rastreamento do MLflow
    mlflow.set_tracking_uri("file://" + os.path.abspath("./mlruns"))
    mlflow.set_experiment("Acidentes de Trânsito - TomekLinks + Class Weights + RandomizedSearch")

    # Carregar os dados
    X_train_raw, X_test_raw, y_train, y_test = load_data()
    preprocessor = load_preprocessor()

    # Aplicar o UnderSampler
    # Você pode alternar entre as técnicas comentando/descomentando
    # X_train_resampled, y_train_resampled = apply_sampling('ClusterCentroids', X_train_raw, y_train, preprocessor)
    # X_train_resampled, y_train_resampled = apply_sampling('NearMiss', X_train_raw, y_train, preprocessor)
    X_train_resampled, y_train_resampled = apply_sampling('TomekLinks', X_train_raw, y_train, preprocessor)
    # X_train_resampled, y_train_resampled = apply_sampling('RandomUnderSampler', X_train_raw, y_train, preprocessor)

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

    # Definir os espaços de hiperparâmetros
    param_distributions_rf = {
        'n_estimators': [50, 100],
        'max_depth': [None, 5],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'bootstrap': [True, False]
    }

    param_distributions_lr = {
        'C': [0.1, 1],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga'],
        'max_iter': [500, 1000]
    }

    param_distributions_xgb = {
        'n_estimators': [50, 100],
        'max_depth': [3, 5],
        'learning_rate': [0.05, 0.1]
    }

    param_distributions_lgb = {
        'n_estimators': [50],
        'num_leaves': [15, 31],
        'max_depth': [3, 5],
        'learning_rate': [0.1],
        'min_data_in_leaf': [30, 50],
        'feature_fraction': [0.8],
        'bagging_fraction': [0.8],
        'bagging_freq': [5],
    }

    param_distributions_cb = {
        'iterations': [50, 100],
        'depth': [4, 6],
        'learning_rate': [0.05, 0.1]
    }

    param_distributions = {
        'Random Forest': param_distributions_rf,
        'Logistic Regression': param_distributions_lr,
        'XGBoost': param_distributions_xgb,
        'LightGBM': param_distributions_lgb,
        'CatBoost': param_distributions_cb
    }

    for model_name, model in models.items():
        print(f"\nTreinando o modelo: {model_name}")

        # Realizar o ajuste de hiperparâmetros
        print("Iniciando o ajuste de hiperparâmetros...")
        params = param_distributions.get(model_name, {})
        
        if model_name == 'LightGBM':
            best_model, best_params = perform_randomized_search_lgb(
                model,
                params,
                X_train_resampled,
                y_train_resampled,
                class_weights,
                n_iter=5,
                cv=2
            )
        else:
            best_model, best_params = perform_randomized_search(
                model,
                params,
                X_train_resampled,
                y_train_resampled,
                n_iter=5,
                cv=2
            )

        # Treinar e avaliar o modelo com os melhores hiperparâmetros
        accuracy, report, cm_filename = train_evaluate_model(
            best_model, X_train_resampled, y_train_resampled, X_test_raw, y_test,
            model_name, preprocessor, data_preprocessed=True
        )

        # Preparar parâmetros e métricas para o MLflow
        params = {
            "sampling": "TomekLinks",
            "class_weight": class_weights,
            "model_name": model_name,
            **best_params  # Inclui os melhores hiperparâmetros
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
            best_model,
            preprocessor,
            run_name=f"{model_name} com TomekLinks e Class Weights - Hyperparameter Tuning"
        )

if __name__ == "__main__":
    train_model()
