# trainBCW.py

import os
import mlflow
from train_utils import (
    train_evaluate_model, load_data, load_preprocessor, log_mlflow,
    perform_randomized_search
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
    mlflow.set_experiment("Balanceamento + Pesos nas Classes")

    # Carregar os dados
    X_train_raw, X_test_raw, y_train, y_test = load_data()
    preprocessor = load_preprocessor()

    # Definir os pesos das classes manualmente
    class_weights_dict = {0: 1, 1: 1, 2: 5}
    print("Pesos das classes:", class_weights_dict)

    # Preprocessar os dados de treinamento
    X_train = preprocessor.transform(X_train_raw)

    # Definir os modelos
    models = {
        'Random Forest': RandomForestClassifier(random_state=42, n_jobs=-1),
        'Logistic Regression': LogisticRegression(max_iter=5000, random_state=42),
        'XGBoost': xgb.XGBClassifier(eval_metric='mlogloss', random_state=42, use_label_encoder=False, n_jobs=-1),
        'LightGBM': lgb.LGBMClassifier(random_state=42, n_jobs=-1),
        'CatBoost': CatBoostClassifier(random_state=42, verbose=0, thread_count=-1)
    }

    # Definir os hiperparâmetros para a busca
    # [As mesmas definições de hiperparâmetros que você já tinha]

    # Dicionário de distribuições de hiperparâmetros
    param_distributions = {
        'Random Forest': {
            'classifier__n_estimators': [50, 100],
            'classifier__max_depth': [None, 5],
            'classifier__min_samples_split': [2, 5],
            'classifier__min_samples_leaf': [1, 2],
            'classifier__bootstrap': [True, False]
        },
        'Logistic Regression': {
            'classifier__C': [0.1, 1],
            'classifier__penalty': ['l1', 'l2'],
            'classifier__solver': ['liblinear', 'saga'],
            'classifier__max_iter': [500, 1000]
        },
        'XGBoost': {
            'classifier__n_estimators': [50, 100],
            'classifier__max_depth': [3, 5],
            'classifier__learning_rate': [0.05, 0.1]
        },
        'LightGBM': {
            'classifier__n_estimators': [50, 100],
            'classifier__num_leaves': [15, 31],
            'classifier__max_depth': [3, 5],
            'classifier__learning_rate': [0.05, 0.1],
            'classifier__min_data_in_leaf': [30, 50],
            'classifier__feature_fraction': [0.8],
            'classifier__bagging_fraction': [0.8],
            'classifier__bagging_freq': [5],
        },
        'CatBoost': {
            'classifier__iterations': [50, 100],
            'classifier__depth': [4, 6],
            'classifier__learning_rate': [0.05, 0.1]
        }
    }

    # Loop sobre os modelos
    for model_name, model in models.items():
        print(f"\nTreinando o modelo: {model_name}")

        # Clonar o modelo
        model_clone = clone(model)

        # Configurar pesos de classe
        if hasattr(model_clone, 'class_weight'):
            model_clone.set_params(class_weight=class_weights_dict)
            use_sample_weight = False
        elif model_name == 'CatBoost':
            model_clone.set_params(class_weights=list(class_weights_dict.values()))
            use_sample_weight = False
        elif model_name == 'LightGBM':
            model_clone.set_params(class_weight=class_weights_dict)
            use_sample_weight = False
        elif model_name == 'XGBoost':
            # XGBoost não suporta class_weight em problemas multiclasse, então usaremos sample_weight
            use_sample_weight = True
        else:
            use_sample_weight = True  # Para modelos que não suportam class_weight

        # Construir o pipeline
        steps = [('classifier', model_clone)]
        model_pipeline = Pipeline(steps)

        # Definir hiperparâmetros
        params = param_distributions.get(model_name, {})

        # Gerar sample_weight se necessário
        if use_sample_weight:
            sample_weight_train = np.array([class_weights_dict[label] for label in y_train])
        else:
            sample_weight_train = None

        # Realizar a busca aleatória de hiperparâmetros
        print("Iniciando o ajuste de hiperparâmetros...")
        best_model, best_params = perform_randomized_search(
            model_pipeline,
            params,
            X_train,
            y_train,
            n_iter=5,
            cv=3,
            sample_weight=sample_weight_train
        )

        # Treinar e avaliar o modelo
        metrics_dict, cm_filename, report_str = train_evaluate_model(
            best_model, X_train, y_train, X_test_raw, y_test,
            model_name, sampling=None, preprocessor=preprocessor, data_preprocessed=True,
            sample_weight=sample_weight_train
        )

        # Preparar parâmetros e métricas para o MLflow
        params = {
            "model_name": model_name,
            **best_params,
            "class_weights": class_weights_dict
        }
        metrics = metrics_dict
        artifacts = {
            "confusion_matrix": cm_filename
        }

        # Registrar no MLflow
        run_name = f"{model_name} - Custom Class Weights"
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
