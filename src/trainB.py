# trainB.py

import os
import mlflow
from train_utils import (
    train_evaluate_model, load_data, load_preprocessor, log_mlflow,
    perform_randomized_search, apply_sampling
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.pipeline import Pipeline

def train_model():
    from sklearn.base import clone
    # Configure MLflow
    mlflow.set_tracking_uri("file://" + os.path.abspath("./mlruns"))
    mlflow.set_experiment("Acidentes de Trânsito - Balanceamento sem Pesos nas Classes")

    # Load data
    X_train_raw, X_test_raw, y_train, y_test = load_data()
    preprocessor = load_preprocessor()

    # Define sampling techniques
    sampling_techniques = ['RandomUnderSampler', 'SMOTE', 'SMOTEENN', 'TomekLinks']

    # Define models
    models = {
        'Random Forest': RandomForestClassifier(random_state=42, n_jobs=-1),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'XGBoost': xgb.XGBClassifier(eval_metric='mlogloss', random_state=42, use_label_encoder=False, n_jobs=-1),
        'LightGBM': lgb.LGBMClassifier(random_state=42, n_jobs=-1),
        'CatBoost': CatBoostClassifier(random_state=42, verbose=0, thread_count=-1)
    }

    # Hyperparameter distributions
    # Hyperparameter distributions
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
        'classifier__max_iter': [2000, 5000]
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

    # Loop over sampling techniques
    for sampling in sampling_techniques:
        print(f"\nTécnica de balanceamento: {sampling}")

        # Apply sampling
        X_resampled, y_resampled, distribution_plot_path, _  = apply_sampling(sampling, X_train_raw, y_train, preprocessor)

        # Loop over models
        for model_name, model in models.items():
            print(f"\nTreinando o modelo: {model_name}")

            # Clone the model
            model_clone = clone(model)

            # Do not apply class weights after resampling
            if hasattr(model_clone, 'class_weight'):
                model_clone.set_params(class_weight=None)

            # Build the pipeline
            steps = [('classifier', model_clone)]
            model_pipeline = Pipeline(steps)

            # Define hyperparameters
            params = param_distributions.get(model_name, {})

            # Perform hyperparameter tuning
            print("Iniciando o ajuste de hiperparâmetros...")
            best_model, best_params = perform_randomized_search(
                model_pipeline,
                params,
                X_resampled,
                y_resampled,
                n_iter=5,
                cv=3
            )

            # Train and evaluate the model
            metrics_dict, cm_filename, report_str = train_evaluate_model(
                best_model, X_resampled, y_resampled, X_test_raw, y_test,
                model_name, sampling=sampling, preprocessor=preprocessor, data_preprocessed=True
            )

            # Prepare parameters and metrics for MLflow
            params = {
                "sampling": sampling,
                "model_name": model_name,
                **best_params
            }
            metrics = metrics_dict
            artifacts = {
                "confusion_matrix": cm_filename,
                "class_distribution": distribution_plot_path
            }

            # Log to MLflow
            run_name = f"{model_name} - {sampling} - No Class Weights"
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
