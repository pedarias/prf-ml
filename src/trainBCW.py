# train.py

import os
import mlflow
from train_utils import (
    train_evaluate_model, get_sampler, load_data, load_preprocessor, log_mlflow,
    perform_randomized_search, apply_sampling
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.pipeline import Pipeline

import numpy as np

def train_model():
    from sklearn.base import clone
    # Configurar o MLflow
    mlflow.set_tracking_uri("file://" + os.path.abspath("./mlruns"))
    mlflow.set_experiment("Acidentes de Trânsito - Balanceamento e Pesos nas Classes")

    # Carregar os dados
    X_train_raw, X_test_raw, y_train, y_test = load_data()
    preprocessor = load_preprocessor()

    # Definir as técnicas de balanceamento
    sampling_techniques = [None, 'RandomUnderSampler', 'SMOTE', 'SMOTEENN', 'TomekLinks']

    # Definir os modelos
    models = {
        'Random Forest': RandomForestClassifier(random_state=42, n_jobs=-1),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1),
        'XGBoost': xgb.XGBClassifier(eval_metric='mlogloss', random_state=42, use_label_encoder=False, n_jobs=-1),
        'LightGBM': lgb.LGBMClassifier(random_state=42, n_jobs=-1),
        'CatBoost': CatBoostClassifier(random_state=42, verbose=0, thread_count=-1)
    }

    # Definir os hiperparâmetros para a busca
    param_distributions_rf = {
        'classifier__n_estimators': [50, 100],
        'classifier__max_depth': [None, 5],
        'classifier__min_samples_split': [2, 5],
        'classifier__min_samples_leaf': [1, 2],
        'classifier__bootstrap': [True, False]
    }

    param_distributions_lr = {
        'classifier__C': [0.1, 1],
        'classifier__penalty': ['l1', 'l2'],
        'classifier__solver': ['liblinear', 'saga'],
        'classifier__max_iter': [500, 1000]
    }

    param_distributions_xgb = {
        'classifier__n_estimators': [50, 100],
        'classifier__max_depth': [3, 5],
        'classifier__learning_rate': [0.05, 0.1]
    }

    param_distributions_lgb = {
        'classifier__n_estimators': [50, 100],
        'classifier__num_leaves': [15, 31],
        'classifier__max_depth': [3, 5],
        'classifier__learning_rate': [0.05, 0.1],
        'classifier__min_data_in_leaf': [30, 50],
        'classifier__feature_fraction': [0.8],
        'classifier__bagging_fraction': [0.8],
        'classifier__bagging_freq': [5],
    }

    param_distributions_cb = {
        'classifier__iterations': [50, 100],
        'classifier__depth': [4, 6],
        'classifier__learning_rate': [0.05, 0.1]
    }

    param_distributions = {
        'Random Forest': param_distributions_rf,
        'Logistic Regression': param_distributions_lr,
        'XGBoost': param_distributions_xgb,
        'LightGBM': param_distributions_lgb,
        'CatBoost': param_distributions_cb
    }

    # Loop sobre as técnicas de balanceamento
    for sampling in sampling_techniques:
        print(f"\nTécnica de balanceamento: {sampling}")

        # Aplicar o balanceamento
        X_resampled, y_resampled, distribution_plot_path, class_counts = apply_sampling(sampling, X_train_raw, y_train, preprocessor)

        # Definir pesos manuais após o balanceamento
        class_weights = {0: 1, 1: 1, 2: 5}  # Dando mais peso à classe 'Com Vítimas Fatais'

        # Loop sobre os modelos
        for model_name, model in models.items():
            print(f"\nTreinando o modelo: {model_name}")

            # Clonar o modelo para evitar interferência entre iterações
            model_clone = clone(model)

            # Configurar pesos de classe
            if hasattr(model_clone, 'class_weight'):
                model_clone.set_params(class_weight=class_weights)
                use_sample_weight = False
            elif model_name == 'CatBoost':
                # Para CatBoost, usamos o parâmetro 'class_weights'
                model_clone.set_params(class_weights=list(class_weights.values()))
                use_sample_weight = False
            elif model_name == 'XGBoost':
                # Para XGBoost, usamos 'scale_pos_weight' para a classe minoritária
                model_clone.set_params(scale_pos_weight=class_weights[2])
                use_sample_weight = False
            else:
                # Para modelos que não suportam class_weight, usaremos sample_weight
                use_sample_weight = True

            # Construir o pipeline (sem o sampler)
            steps = []
            steps.append(('classifier', model_clone))

            model_pipeline = Pipeline(steps)

            # Definir os hiperparâmetros para a busca
            params = param_distributions.get(model_name, {})

            # Realizar a busca aleatória de hiperparâmetros
            print("Iniciando o ajuste de hiperparâmetros...")
            best_model, best_params = perform_randomized_search(
                model_pipeline,
                params,
                X_resampled,
                y_resampled,
                n_iter=5,
                cv=3
            )

            # Gerar sample_weight se necessário
            if use_sample_weight:
                sample_weight_train = np.array([class_weights[label] for label in y_resampled])
            else:
                sample_weight_train = None

            # Treinar e avaliar o modelo com os melhores hiperparâmetros
            metrics_dict, cm_filename, report_str = train_evaluate_model(
                best_model, X_resampled, y_resampled, X_test_raw, y_test,
                model_name, sampling=sampling, preprocessor=preprocessor, data_preprocessed=True,
                sample_weight=sample_weight_train
            )

            # Preparar parâmetros e métricas para o MLflow
            params = {
                "sampling": sampling,
                "model_name": model_name,
                **best_params
            }
            metrics = metrics_dict
            artifacts = {
                "confusion_matrix": cm_filename,
                "class_distribution": distribution_plot_path  # Adiciona o gráfico de distribuição
            }

            # Registrar no MLflow
            run_name = f"{model_name} - {sampling or 'No Sampling'} - Class Weights"
            log_mlflow(
                model_name,
                params,
                metrics,
                artifacts,
                best_model,
                run_name=run_name,
                classification_report_str=report_str
            )

if __name__ == "__main__":
    train_model()
