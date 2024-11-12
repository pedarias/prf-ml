# trainCW.py

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

from sklearn.utils.class_weight import compute_class_weight
import numpy as np

def train_model():
    from sklearn.base import clone
    # Configure MLflow
    mlflow.set_tracking_uri("file://" + os.path.abspath("./mlruns"))
    mlflow.set_experiment("Acidentes de Trânsito - Pesos nas Classes")

    # Load data
    X_train_raw, X_test_raw, y_train, y_test = load_data()
    preprocessor = load_preprocessor()

    # Compute class weights
    classes = np.unique(y_train)
    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weights_dict = dict(zip(classes, class_weights))
    print("Pesos das classes:", class_weights_dict)

    # Apply sampling (none in this case) to keep consistency and visualize class distribution
    sampling = None  # No sampling
    X_resampled, y_resampled, distribution_plot_path, _ = apply_sampling(
        sampling_technique=sampling,
        X_train_raw=X_train_raw,
        y_train=y_train,
        preprocessor=preprocessor,
        data_preprocessed=False,
        preprocess_before_sampling=True
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
            'classifier__n_estimators': [50, 100],
            'classifier__max_depth': [3, 6, 9],
            'classifier__learning_rate': [0.1, 0.01]
        },
        'LightGBM': {
            'classifier__n_estimators': [50, 100],
            'classifier__num_leaves': [31],
            'classifier__max_depth': [3,6,9],
            'classifier__learning_rate': [0.1],
            'classifier__min_data_in_leaf': [30],
            'classifier__feature_fraction': [0.8],
            'classifier__bagging_fraction': [0.8],
            'classifier__bagging_freq': [5],
        },
        'CatBoost': {
            'classifier__iterations': [50, 100],
            'classifier__depth': [3,6,9],
            'classifier__learning_rate': [0.1, 0.01]
        }
    }

    # Loop over models
    for model_name, model in models.items():
        print(f"\nTraining model: {model_name}")

        model_clone = clone(model)

        # Configure class weights
        if hasattr(model_clone, 'class_weight'):
            model_clone.set_params(class_weight=class_weights_dict)
            use_sample_weight = False
        elif model_name == 'CatBoost':
            model_clone.set_params(class_weights=list(class_weights))
            use_sample_weight = False
        elif model_name == 'LightGBM':
            model_clone.set_params(class_weight=class_weights_dict)
            use_sample_weight = False
        elif model_name == 'XGBoost':
            # XGBoost does not support class_weight in multiclass problems, so we'll use sample_weight
            use_sample_weight = True
        else:
            use_sample_weight = True  # For models that require sample_weight

        steps = [('classifier', model_clone)]
        model_pipeline = Pipeline(steps)

        param_grid = param_grids.get(model_name, {})

        if use_sample_weight:
            sample_weight_train = np.array([class_weights_dict[label] for label in y_resampled])
        else:
            sample_weight_train = None

        print("Initiating hyperparameter tuning...")
        best_model, best_params = perform_grid_search(
            model_pipeline,
            param_grid,
            X_resampled,
            y_resampled,
            cv=3,
            sample_weight=sample_weight_train
        )

        metrics_dict, cm_filename, report_str = train_evaluate_model(
            best_model, X_resampled, y_resampled, X_test_raw, y_test,
            model_name, sampling=sampling, preprocessor=preprocessor,
            data_preprocessed_train=True,    # Training data is preprocessed
            data_preprocessed_test=False,    # Testing data is raw
            sample_weight=sample_weight_train
        )

        # Prepare parameters and metrics for MLflow
        params = {
            "model_name": model_name,
            **best_params,
            "class_weights": class_weights_dict
        }
        metrics = metrics_dict
        artifacts = {
            "confusion_matrix": cm_filename,
            "class_distribution": distribution_plot_path
        }

        # Log to MLflow
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
