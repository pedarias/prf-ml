#src/register_promote_model

import mlflow
from mlflow.tracking import MlflowClient

def register_and_promote_model(run_id, model_name, production_model_name):
    # Registrar o Modelo
    model_uri = f'runs:/{run_id}/model'
    with mlflow.start_run(run_id=run_id):
        mlflow.register_model(model_uri=model_uri, name=model_name)
    
    # Promover o Modelo para Produção
    client = MlflowClient()
    current_model_uri = f"models:/{model_name}/1"
    client.copy_model_version(src_model_uri=current_model_uri, dst_name=production_model_name)

if __name__ == "__main__":
    run_id = input('Please type RunID: ')
    model_name = 'XGB-Smote-ClassWeights'
    production_model_name = "prfml-prod"
    
    register_and_promote_model(run_id, model_name, production_model_name)
