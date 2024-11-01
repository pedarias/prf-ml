# src/register_promote_model.py

import mlflow
from mlflow.tracking import MlflowClient

def register_and_promote_best_model(experiment_name, metric_name, minimum_metric_threshold, model_name, production_model_name):
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    
    if experiment is None:
        print(f"Experimento '{experiment_name}' não encontrado.")
        return

    # Obter todas as execuções do experimento
    runs = client.search_runs(
        experiment_ids=experiment.experiment_id,
        filter_string="",
        run_view_type=mlflow.entities.ViewType.ACTIVE_ONLY,
        max_results=1000
    )

    # Filtrar runs que possuem a métrica desejada e que atendem ao threshold mínimo
    valid_runs = []
    for run in runs:
        metrics = run.data.metrics
        if metric_name in metrics and metrics[metric_name] >= minimum_metric_threshold:
            valid_runs.append(run)

    if not valid_runs:
        print("Nenhum modelo atende aos critérios especificados.")
        return

    # Selecionar o modelo com a melhor métrica
    best_run = max(valid_runs, key=lambda run: run.data.metrics[metric_name])
    best_run_id = best_run.info.run_id
    best_metric_value = best_run.data.metrics[metric_name]

    print(f"Melhor modelo encontrado: Run ID = {best_run_id}, {metric_name} = {best_metric_value}")

    # Registrar o Modelo
    model_uri = f"runs:/{best_run_id}/model"
    model_version = mlflow.register_model(model_uri=model_uri, name=model_name)
    
    # Promover o Modelo para Produção
    client.transition_model_version_stage(
        name=model_name,
        version=model_version.version,
        stage="Production",
        archive_existing_versions=True
    )

    print(f"Modelo '{model_name}' versão '{model_version.version}' promovido para produção.")

if __name__ == "__main__":
    experiment_name = "Acidentes de Trânsito - Comparação de Técnicas de Balanceamento"
    metric_name = "f1_score_fatal"
    minimum_metric_threshold = 0.5  # Defina o valor mínimo desejado
    model_name = 'LightGBM - SMOTE - No Class Weights'
    production_model_name = "prfml-prod"
    
    register_and_promote_best_model(experiment_name, metric_name, minimum_metric_threshold, model_name, production_model_name)
