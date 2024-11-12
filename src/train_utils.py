# src/train_utils.py

import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend to avoid tkinter errors

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import tempfile
import os
import joblib
import pickle
from sklearn.metrics import (
    classification_report, accuracy_score, confusion_matrix,
    make_scorer, f1_score, roc_auc_score
)
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import (
    RandomUnderSampler, NearMiss, TomekLinks, ClusterCentroids
)
from sklearn.base import clone

def visualize_class_distribution(y, title='Distribuição das Classes', save_path=None):
    class_counts = pd.Series(y).value_counts().sort_index()
    class_labels = ['Sem Vítimas', 'Com Vítimas Feridas', 'Com Vítimas Fatais']

    plt.figure(figsize=(8, 6))
    sns.barplot(x=class_labels, y=class_counts.values, palette='viridis')
    plt.title(title)
    plt.xlabel('Classe')
    plt.ylabel('Número de Ocorrências')
    plt.xticks(rotation=15)

    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

    total = class_counts.sum()
    percentages = (class_counts / total) * 100
    for label, count, percentage in zip(class_labels, class_counts, percentages):
        print(f"{label}: {count} ocorrências ({percentage:.2f}%)")

def apply_sampling(sampling_technique, X_train_raw, y_train, preprocessor=None, data_preprocessed=False, preprocess_before_sampling=True):
    """
    Applies sampling techniques to balance the dataset.

    Parameters:
    - sampling_technique: The sampling technique to use.
    - X_train_raw: Raw training data (before preprocessing).
    - y_train: Training labels.
    - preprocessor: The preprocessing pipeline to apply. If None, no preprocessing is applied.
    - data_preprocessed: Boolean indicating if X_train_raw is already preprocessed.
    - preprocess_before_sampling: If True, preprocessing is applied before sampling.

    Returns:
    - X_resampled: Resampled (and possibly preprocessed) training data.
    - y_resampled: Resampled training labels.
    - distribution_plot_path: Path to the saved class distribution plot.
    - class_counts: Class counts after resampling.
    """
    # Determine if preprocessing is needed before sampling
    if not data_preprocessed and preprocessor is not None:
        if preprocess_before_sampling:
            X_processed = preprocessor.transform(X_train_raw)
        else:
            X_processed = X_train_raw
    else:
        X_processed = X_train_raw  # Data is already preprocessed

    sampler = get_sampler(sampling_technique)

    if sampler is not None:
        X_resampled_raw, y_resampled = sampler.fit_resample(X_processed, y_train)

        # Preprocess after sampling if specified
        if not data_preprocessed and preprocessor is not None and not preprocess_before_sampling:
            X_resampled = preprocessor.transform(X_resampled_raw)
        else:
            X_resampled = X_resampled_raw

        # Visualize the class distribution after sampling
        distribution_plot_path = f"class_distribution_{sampling_technique or 'No_Sampling'}.png"
        visualize_class_distribution(y_resampled, title=f'Distribuição Após {sampling_technique}', save_path=distribution_plot_path)
    else:
        X_resampled = X_processed
        y_resampled = y_train
        # Visualize the class distribution without sampling
        distribution_plot_path = f"class_distribution_No_Sampling.png"
        visualize_class_distribution(y_resampled, title='Distribuição Sem Balanceamento', save_path=distribution_plot_path)

    return X_resampled, y_resampled, distribution_plot_path, pd.Series(y_resampled).value_counts()

def train_evaluate_model(model_pipeline, X_train, y_train, X_test_raw, y_test, model_name, sampling, preprocessor, data_preprocessed_train=True, data_preprocessed_test=False, sample_weight=None):
    """
    Trains and evaluates a model pipeline, including preprocessing and balancing.

    Parameters:
    - model_pipeline: The machine learning pipeline to train.
    - X_train: Training data (preprocessed if data_preprocessed_train=True).
    - y_train: Training labels.
    - X_test_raw: Testing data (raw if data_preprocessed_test=False).
    - y_test: Testing labels.
    - model_name: Name of the model (for logging and saving artifacts).
    - sampling: The sampling technique used (for naming purposes).
    - preprocessor: The preprocessing pipeline to apply.
    - data_preprocessed_train: Boolean indicating if X_train is already preprocessed.
    - data_preprocessed_test: Boolean indicating if X_test_raw is already preprocessed.
    - sample_weight: Sample weights for each training instance (optional).

    Returns:
    - metrics_dict: Dictionary containing evaluation metrics.
    - cm_filename: Filename of the saved confusion matrix plot.
    - report_str: String representation of the classification report.
    """

    # Preprocess training data if needed
    if not data_preprocessed_train and preprocessor is not None:
        X_train = preprocessor.transform(X_train)
    
    # Preprocess testing data if needed
    if not data_preprocessed_test and preprocessor is not None:
        X_test = preprocessor.transform(X_test_raw)
    else:
        X_test = X_test_raw

    # Fit the model
    if sample_weight is not None:
        model_pipeline.fit(X_train, y_train, classifier__sample_weight=sample_weight)
    else:
        model_pipeline.fit(X_train, y_train)

    # Make predictions
    y_pred = model_pipeline.predict(X_test)
    y_proba = model_pipeline.predict_proba(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr')
    report = classification_report(
        y_test, y_pred,
        target_names=['Sem Vítimas', 'Com Vítimas Feridas', 'Com Vítimas Fatais'],
        output_dict=True
    )
    # Obtain the classification report as a string
    report_str = classification_report(
        y_test, y_pred,
        target_names=['Sem Vítimas', 'Com Vítimas Feridas', 'Com Vítimas Fatais']
    )

    # Print evaluation metrics
    print(f"Modelo: {model_name}")
    print("Accuracy:", accuracy)
    print("ROC AUC:", roc_auc)
    print("\nClassification Report:")
    print(report_str)

    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Sem Vítimas', 'Com Vítimas Feridas', 'Com Vítimas Fatais'],
                yticklabels=['Sem Vítimas', 'Com Vítimas Feridas', 'Com Vítimas Fatais'])
    plt.xlabel('Predito')
    plt.ylabel('Verdadeiro')
    plt.title(f'Matriz de Confusão - {model_name} - {sampling}')
    # Save the confusion matrix plot
    cm_filename = f"confusion_matrix_{model_name.replace(' ', '_')}_{sampling or 'No_Sampling'}.png"
    plt.savefig(cm_filename)
    plt.close()

    # Compile metrics into a dictionary
    metrics_dict = {
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'f1_score_fatal': report['Com Vítimas Fatais']['f1-score'],
        'recall_fatal': report['Com Vítimas Fatais']['recall'],
        'precision_fatal': report['Com Vítimas Fatais']['precision']
    }

    return metrics_dict, cm_filename, report_str

def get_sampler(sampling_technique):
    """
    Returns the sampler object based on the sampling technique specified.

    Parameters:
    - sampling_technique: The sampling technique to use.

    Returns:
    - sampler: The sampler object or None.
    """
    samplers = {
        'SMOTE': SMOTE(random_state=42),
        'ADASYN': ADASYN(random_state=42),
        'SMOTEENN': SMOTEENN(random_state=42),
        'RandomUnderSampler': RandomUnderSampler(random_state=42),
        'NearMiss': NearMiss(version=1),
        'TomekLinks': TomekLinks(),
        'ClusterCentroids': ClusterCentroids(random_state=42),
        None: None  # No sampling
    }
    return samplers.get(sampling_technique, None)

def f1_fatal(y_true, y_pred):
    return f1_score(y_true, y_pred, labels=[2], average='macro')

def perform_grid_search(estimator, param_grid, X, y, cv=3, scoring=None, sample_weight=None):
    """
    Perform a grid search with cross-validation.

    Parameters:
    - estimator: The pipeline or estimator to tune.
    - param_grid: The parameter grid to explore.
    - X, y: Training data and labels.
    - cv: Cross-validation folds.
    - scoring: Scoring function to optimize.
    - sample_weight: Array of weights that are assigned to individual samples.

    Returns:
    - best_model: The best model from the search.
    - best_params: The best parameters found.
    """
    if scoring is None:
        scoring = 'accuracy'  # Default to accuracy if no scoring function provided

    # Prepare fit parameters
    fit_params = {'classifier__sample_weight': sample_weight} if sample_weight is not None else {}

    grid_search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X, y, **fit_params)
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    return best_model, best_params

def perform_randomized_search(estimator, param_distributions, X, y, n_iter=10, cv=3, scoring=None, sample_weight=None):
    """
    Perform a randomized search with cross-validation.

    Parameters:
    - estimator: The pipeline or estimator to tune.
    - param_distributions: The parameter distributions to explore.
    - X, y: Training data and labels.
    - n_iter, cv: Number of iterations and cross-validation folds.
    - scoring: Scoring function to optimize.
    - sample_weight: Array of weights that are assigned to individual samples.

    Returns:
    - best_model: The best model from the search.
    - best_params: The best parameters found.
    """
    if scoring is None:
        scoring = 'accuracy'  # Default to accuracy if no scoring function provided

    # Prepare fit parameters
    fit_params = {'classifier__sample_weight': sample_weight} if sample_weight is not None else {}

    randomized_search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        random_state=42,
        verbose=1
    )

    randomized_search.fit(X, y, **fit_params)
    best_model = randomized_search.best_estimator_
    best_params = randomized_search.best_params_
    
    return best_model, best_params

# Define a custom MLflow PyFunc model class
class AccidentSeverityModel(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        import joblib
        # Load the preprocessing pipeline
        self.preprocessor = joblib.load(context.artifacts["preprocessor"])
        # Load the trained model
        self.model = joblib.load(context.artifacts["model"])

    def predict(self, context, model_input):
        # Apply preprocessing
        processed_input = self.preprocessor.transform(model_input)
        # Make predictions
        return self.model.predict(processed_input)

def load_data():
    # Function to load the data
    X_train_raw = pd.read_csv('../data/processed/X_train_raw.csv', sep='\t')
    X_test_raw = pd.read_csv('../data/processed/X_test_raw.csv', sep='\t')
    y_train = pd.read_csv('../data/processed/train_labels.csv').values.ravel()
    y_test = pd.read_csv('../data/processed/test_labels.csv').values.ravel()
    return X_train_raw, X_test_raw, y_train, y_test

def load_preprocessor():
    with open('../artifacts/preprocessor.pkl', 'rb') as f:
        preprocessor = pickle.load(f)
    return preprocessor

def log_mlflow(model_name, params, metrics, artifacts, model_pipeline, run_name, classification_report_str=None):
    with mlflow.start_run(run_name=run_name):
        # Log parameters
        mlflow.log_params(params)
        
        # Log metrics
        mlflow.log_metrics(metrics)
        
        # Log artifacts (only those that are file paths)
        for artifact_path in artifacts.values():
            if isinstance(artifact_path, str) and os.path.exists(artifact_path):
                mlflow.log_artifact(artifact_path)
        
        # Save the classification report as an artifact
        if classification_report_str is not None:
            report_path = os.path.join("classification_report.txt")
            with open(report_path, 'w') as f:
                f.write(classification_report_str)
            mlflow.log_artifact(report_path)
            os.remove(report_path)
        
        # Save the complete model as an artifact
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = os.path.join(tmp_dir, "model_pipeline.pkl")
            preprocessor_path = os.path.join(tmp_dir, "preprocessor.pkl")
            joblib.dump(model_pipeline, model_path)
            joblib.dump(load_preprocessor(), preprocessor_path)
            # Log the model with MLflow
            mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=AccidentSeverityModel(),
                artifacts={
                    "model": model_path,
                    "preprocessor": preprocessor_path
                }
            )
