# train.py

import os
import mlflow
from train_utils import (
    train_evaluate_model, load_data, load_preprocessor, log_mlflow,
    perform_grid_search, apply_sampling
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
    mlflow.set_experiment("Acidentes de Trânsito - Sem Pesos ou Balanceamento")

    # Load data
    X_train_raw, X_test_raw, y_train, y_test = load_data()
    preprocessor = load_preprocessor()

    # No sampling techniques
    sampling_techniques = [None]

    # Loop over the sampling techniques (only None in this case)
    for sampling in sampling_techniques:
        print(f"\nTécnica de balanceamento: {sampling}")

        # Apply sampling (which will not alter the data in this case)
        X_resampled, y_resampled, distribution_plot_path, _ = apply_sampling(
            sampling_technique=sampling,
            X_train_raw=X_train_raw,
            y_train=y_train,
            preprocessor=preprocessor,
            data_preprocessed=False,  # Data is not preprocessed yet
            preprocess_before_sampling=True  # Preprocess before sampling
        )

        # Define models
        models = {
            'Random Forest': RandomForestClassifier(random_state=42, n_jobs=-1),
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'XGBoost': xgb.XGBClassifier(eval_metric='mlogloss', random_state=42, use_label_encoder=False, n_jobs=-1),
            'LightGBM': lgb.LGBMClassifier(random_state=42, n_jobs=-1),
            'CatBoost': CatBoostClassifier(random_state=42, verbose=0, thread_count=-1)
        }

        # Hyperparameter grids
        param_grids = {
            'Random Forest': {
                'classifier__n_estimators': [50, 100],
                'classifier__max_depth': [None, 5],
                'classifier__min_samples_split': [2],
                'classifier__min_samples_leaf': [1],
                'classifier__bootstrap': [True]
            },
            'Logistic Regression': {
                'classifier__C': [0.1, 1],
                'classifier__penalty': ['l2'],
                'classifier__solver': ['liblinear'],
                'classifier__max_iter': [500]
            },
            'XGBoost': {
                'classifier__n_estimators': [500, 1000],
                'classifier__max_depth': [3, 5],
                'classifier__learning_rate': [0.1]
            },
            'LightGBM': {
                'classifier__n_estimators': [500, 1000],
                'classifier__num_leaves': [31],
                'classifier__max_depth': [5],
                'classifier__learning_rate': [0.1],
                'classifier__min_data_in_leaf': [30],
                'classifier__feature_fraction': [0.8],
                'classifier__bagging_fraction': [0.8],
                'classifier__bagging_freq': [5],
            },
            'CatBoost': {
                'classifier__iterations': [500, 1000],
                'classifier__depth': [6],
                'classifier__learning_rate': [0.1]
            }
        }

        # Loop over the models
        for model_name, model in models.items():
            print(f"\nTreinando o modelo: {model_name}")

            # Clone the model to avoid interference between iterations
            model_clone = clone(model)

            # Do not set class weights
            # Ensure that class_weight is set to None if applicable
            if hasattr(model_clone, 'class_weight'):
                model_clone.set_params(class_weight=None)

            # Build the pipeline (only the classifier since preprocessing is already done)
            steps = []
            steps.append(('classifier', model_clone))

            model_pipeline = Pipeline(steps)

            # Define hyperparameters for the search
            params = param_grids.get(model_name, {})

            # Perform grid search hyperparameter tuning
            print("Iniciando o ajuste de hiperparâmetros...")
            best_model, best_params = perform_grid_search(
                model_pipeline,
                params,
                X_resampled,
                y_resampled,
                cv=3
            )

            # Train and evaluate the model with the best hyperparameters
            metrics_dict, cm_filename, report_str = train_evaluate_model(
                best_model, X_resampled, y_resampled, X_test_raw, y_test,
                model_name, sampling=sampling, preprocessor=preprocessor,
                data_preprocessed_train=True,  # Training data is preprocessed
                data_preprocessed_test=False   # Testing data is raw
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

            # Register in MLflow
            run_name = f"{model_name} - No Weights or Sampling"
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
