# src/train_utils.py

import matplotlib
matplotlib.use('Agg')  # Usar um backend não interativo para evitar erros do tkinter

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
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
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
    sns.barplot(x=class_labels, y=class_counts.values, hue=class_labels, dodge=False, palette='viridis', legend=False)
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


def apply_sampling(sampling_technique, X_train_raw, y_train, preprocessor):
    """
    Applies sampling techniques to balance the dataset.

    Parameters:
    - sampling_technique: The sampling technique to use ('SMOTE', 'ADASYN', 'SMOTEENN', 'RandomUnderSampler', 'NearMiss', 'TomekLinks', 'ClusterCentroids', or None).
    - X_train_raw: Raw training data (before preprocessing).
    - y_train: Training labels.
    - preprocessor: The preprocessing pipeline to apply.

    Returns:
    - X_resampled: Resampled preprocessed training data.
    - y_resampled: Resampled training labels.
    - distribution_plot_path: Path to the saved class distribution plot.
    """
    # Apply preprocessing to the raw data
    X_train = preprocessor.transform(X_train_raw)

    sampler = get_sampler(sampling_technique)

    if sampler is not None:
        X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
        # Visualize the class distribution after sampling
        distribution_plot_path = f"class_distribution_{sampling_technique or 'No_Sampling'}.png"
        visualize_class_distribution(y_resampled, title=f'Distribuição Após {sampling_technique}', save_path=distribution_plot_path)
    else:
        X_resampled, y_resampled = X_train, y_train
        # Visualize the class distribution without sampling
        distribution_plot_path = f"class_distribution_{sampling_technique or 'No_Sampling'}.png"
        visualize_class_distribution(y_resampled, title='Distribuição Sem Balanceamento', save_path=distribution_plot_path)

    return X_resampled, y_resampled, distribution_plot_path, pd.Series(y_resampled).value_counts()

def train_evaluate_model(model_pipeline, X_train, y_train, X_test_raw, y_test, model_name, sampling, preprocessor, data_preprocessed=False, sample_weight=None):
    """
    Trains and evaluates a model pipeline, including preprocessing and balancing.

    Parameters:
    - model_pipeline: The machine learning pipeline to train.
    - X_train: Preprocessed training data.
    - y_train: Training labels.
    - X_test_raw: Raw testing data.
    - y_test: Testing labels.
    - model_name: Name of the model (for logging and saving artifacts).
    - sampling: The sampling technique used (for naming purposes).
    - preprocessor: The preprocessing pipeline to apply.
    - data_preprocessed: Boolean indicating if X_train is already preprocessed.
    - sample_weight: Sample weights for each training instance (optional).

    Returns:
    - metrics_dict: Dictionary containing evaluation metrics.
    - cm_filename: Filename of the saved confusion matrix plot.
    - report_str: String representation of the classification report.
    """

    # Apply the preprocessing pipeline to the testing data
    X_test = preprocessor.transform(X_test_raw)

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

def perform_randomized_search(model_pipeline, param_distributions, X_train, y_train, n_iter=10, cv=3, scoring=None):
    """
    Performs randomized search over hyperparameters with cross-validation.

    Parameters:
    - model_pipeline: The machine learning pipeline.
    - param_distributions: Dictionary with parameters names as keys and distributions or lists of parameters to try.
    - X_train: Preprocessed training data.
    - y_train: Training labels.
    - n_iter: Number of parameter settings that are sampled.
    - cv: Number of folds in StratifiedKFold.
    - scoring: A single string or callable to evaluate the predictions on the test set.

    Returns:
    - best_model: The best model found during the search.
    - best_params: The parameters of the best model.
    """
    if scoring is None:
        scoring = make_scorer(f1_fatal)

    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    randomized_search = RandomizedSearchCV(
        estimator=model_pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv_strategy,
        scoring=scoring,
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    randomized_search.fit(X_train, y_train)
    best_model = randomized_search.best_estimator_
    best_params = randomized_search.best_params_
    return best_model, best_params

# Define a custom MLflow PyFunc model class
class AccidentSeverityModel(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        # Import joblib dentro da função, se necessário
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
    # Função para carregar os dados
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
        # Logar os parâmetros
        mlflow.log_params(params)
        
        # Logar as métricas
        mlflow.log_metrics(metrics)
        
        # Logar os artefatos (apenas aqueles que são caminhos de arquivo)
        for artifact_path in artifacts.values():
            if isinstance(artifact_path, str) and os.path.exists(artifact_path):
                mlflow.log_artifact(artifact_path)
        
        # Salvar o classification_report como um artefato
        if classification_report_str is not None:
            report_path = os.path.join("classification_report.txt")
            with open(report_path, 'w') as f:
                f.write(classification_report_str)
            mlflow.log_artifact(report_path)
            os.remove(report_path)
        
        # Salvar o modelo completo como um artefato
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = os.path.join(tmp_dir, "model_pipeline.pkl")
            preprocessor_path = os.path.join(tmp_dir, "preprocessor.pkl")
            joblib.dump(model_pipeline, model_path)
            joblib.dump(load_preprocessor(), preprocessor_path)
            # Logar o modelo com o MLflow
            mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=AccidentSeverityModel(),
                artifacts={
                    "model": model_path,
                    "preprocessor": preprocessor_path
                }
            )
