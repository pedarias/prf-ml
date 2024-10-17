from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import mlflow.pyfunc
from sklearn.preprocessing import StandardScaler
import json

class Data(BaseModel):
    dia_semana: str
    uf: str
    causa_acidente: str
    tipo_acidente: str
    condicao_metereologica: str
    tipo_pista: str
    tracado_via: str
    PERIODO_DIA: str
    sentido_via: str
    uso_solo: str
    fase_dia: str
    FINAL_DE_SEMANA: int
    HORA: int
    km: float
    veiculos: int
    pessoas: int
    TOTAL_FERIDOS: int

app = FastAPI()

# Carregar o modelo treinado
model_name = "prfml-prod"
model_version = "1"  # Use a versão correta do modelo
model_uri = f"models:/{model_name}/{model_version}"
model = mlflow.pyfunc.load_model(model_uri)

# Carregar as colunas esperadas pelo modelo
with open('columns.json', 'r') as f:
    expected_columns = json.load(f)

# Dicionário de mapeamento de classes
class_mapping = {
    0: "Sem Vítimas",
    1: "Com Vítimas Feridas",
    2: "Com Vítimas Fatais"
}

def prepare_data(input_data):
    # Converter colunas categóricas para o tipo 'category'
    categorical_columns = ['dia_semana', 'uf', 'causa_acidente', 'tipo_acidente', 'condicao_metereologica',
                           'tipo_pista', 'tracado_via', 'PERIODO_DIA', 'sentido_via', 'uso_solo', 'fase_dia']
    for col in categorical_columns:
        input_data[col] = input_data[col].astype('category')
    
    # One-hot encoding
    input_data_encoded = pd.get_dummies(input_data, drop_first=True)
    
    # Padronização
    num_cols = input_data_encoded.select_dtypes(include=['int64', 'float64']).columns.tolist()
    scaler = StandardScaler()
    input_data_encoded[num_cols] = scaler.fit_transform(input_data_encoded[num_cols])
    
    # Garantir que as colunas estejam na mesma ordem e formato que as esperadas pelo modelo
    for col in expected_columns:
        if col not in input_data_encoded.columns:
            input_data_encoded[col] = 0
    input_data_encoded = input_data_encoded[expected_columns]
    
    return input_data_encoded

@app.post("/predict")
def predict(data: Data):
    input_data = pd.DataFrame([data.dict()])
    input_data_prepared = prepare_data(input_data)
    prediction = model.predict(input_data_prepared)
    prediction_class = [class_mapping[pred] for pred in prediction]
    return {"prediction": prediction_class}
