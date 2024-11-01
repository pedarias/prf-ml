# trainCW.py

import os
import mlflow
from train_utils import (
    train_evaluate_model, get_sampler, load_data, load_preprocessor, log_mlflow,
    perform_randomized_search, apply_sampling, f1_fatal
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.pipeline import Pipeline

from sklearn.utils.class_weight import compute_class_weight
import numpy as np

def train_model():
    from sklearn.base import clone
    # Configurar o MLflow
    mlflow.set_tracking_uri("file://" + os.path.abspath("./mlruns"))
    mlflow.set_experiment("Acidentes de Trânsito - Pesos nas Classes")

    # Carregar os dados
    X_train_raw, X_test_raw, y_train, y_test = load_data()
    preprocessor = load_preprocessor()

    # Calcular os pesos das classes
    classes = np.unique(y_train)
    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weights_dict = dict(zip(classes, class_weights))
    print("Pesos das classes:", class_weights_dict)

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

    # Aplicar o pré-processamento aos dados de treinamento
    X_train = preprocessor.transform(X_train_raw)

    # Loop sobre os modelos
    for model_name, model in models.items():
        print(f"\nTreinando o modelo: {model_name}")

        # Clonar o modelo para evitar interferência entre iterações
        model_clone = clone(model)

        # Configurar pesos de classe
        if hasattr(model_clone, 'class_weight'):
            model_clone.set_params(class_weight=class_weights_dict)
            use_sample_weight = False
        elif model_name == 'CatBoost':
            # Para CatBoost, usamos o parâmetro 'class_weights'
            model_clone.set_params(class_weights=list(class_weights))
            use_sample_weight = False
        elif model_name == 'XGBoost':
            # Para XGBoost, usamos 'scale_pos_weight' para a classe minoritária
            scale_pos_weight = class_weights[-1]  # Peso da classe 'Com Vítimas Fatais'
            model_clone.set_params(scale_pos_weight=scale_pos_weight)
            use_sample_weight = False
        else:
            # Para modelos que não suportam class_weight, usaremos sample_weight
            use_sample_weight = True

        # Construir o pipeline (apenas o classificador)
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
            X_train,
            y_train,
            n_iter=5,
            cv=3
        )

        # Treinar e avaliar o modelo com os melhores hiperparâmetros
        if use_sample_weight:
            sample_weight_train = np.array([class_weights_dict[label] for label in y_train])
        else:
            sample_weight_train = None

        metrics_dict, cm_filename, report_str = train_evaluate_model(
            best_model, X_train, y_train, X_test_raw, y_test,
            model_name, sampling=None, preprocessor=preprocessor, data_preprocessed=True,
            sample_weight=sample_weight_train
        )

        # Preparar parâmetros e métricas para o MLflow
        params = {
            "model_name": model_name,
            **best_params
        }
        metrics = {
            "accuracy": metrics_dict['accuracy'],
            "f1_score_fatal": metrics_dict['f1_score_fatal'],
            "recall_fatal": metrics_dict['recall_fatal'],
            "precision_fatal": metrics_dict['precision_fatal']
        }
        artifacts = {
            "confusion_matrix": cm_filename
        }

        # Registrar no MLflow
        run_name = f"{model_name} - Class Weights"
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
