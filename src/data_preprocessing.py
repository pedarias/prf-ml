#src/data_preprocessing.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import json

def preprocess_data(input_path, train_path, test_path):
    df = pd.read_csv(input_path)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    # Conversão de tipos e tratamento de strings
    df['data_inversa'] = pd.to_datetime(df['data_inversa'])
    df['horario'] = pd.to_datetime(df['horario'], format='%H:%M:%S').dt.time
    df['km'] = df['km'].astype(str).str.replace(',', '.').astype(float)
    df['latitude'] = df['latitude'].astype(str).str.replace(',', '.').astype(float)
    df['longitude'] = df['longitude'].astype(str).str.replace(',', '.').astype(float)

    cols_str = ['dia_semana', 'uf', 'municipio', 'causa_acidente', 'tipo_acidente',
                'classificacao_acidente', 'fase_dia', 'sentido_via', 'condicao_metereologica',
                'tipo_pista', 'tracado_via', 'uso_solo', 'regional', 'delegacia', 'uop']
    for col in cols_str:
        df[col] = df[col].astype(str).str.strip().str.upper()

    # Engenharia de features
    df['HORA'] = df['horario'].apply(lambda x: x.hour)
    df['PERIODO_DIA'] = df['HORA'].apply(lambda x: 'MANHÃ' if 5 <= x < 12 else 'TARDE' if 12 <= x < 18 else 'NOITE' if 18 <= x < 24 else 'MADRUGADA')
    df['FINAL_DE_SEMANA'] = df['dia_semana'].apply(lambda x: 1 if x in ['SÁBADO', 'DOMINGO'] else 0)
    df['TOTAL_FERIDOS'] = df['feridos_leves'] + df['feridos_graves']
    gravidade_mapping = {'SEM VÍTIMAS': 0, 'COM VÍTIMAS FERIDAS': 1, 'COM VÍTIMAS FATAIS': 2}
    df['GRAVIDADE'] = df['classificacao_acidente'].map(gravidade_mapping)

    # Seleção de features para o modelo
    features = ['dia_semana', 'uf', 'causa_acidente', 'tipo_acidente',
                'condicao_metereologica', 'tipo_pista', 'tracado_via',
                'PERIODO_DIA', 'sentido_via', 'uso_solo', 'fase_dia',
                'FINAL_DE_SEMANA', 'HORA', 'km', 'veiculos', 'pessoas',
                'TOTAL_FERIDOS']
    X = df[features]
    y = df['GRAVIDADE']

    # One-hot encoding e padronização
    X_encoded = pd.get_dummies(X, drop_first=True)
    num_cols = X_encoded.select_dtypes(include=['int64', 'float64']).columns.tolist()
    scaler = StandardScaler()
    X_encoded[num_cols] = scaler.fit_transform(X_encoded[num_cols])

    # Divisão em conjuntos de treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42, stratify=y)

    # Salvando os dados processados
    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    X_train.to_csv(train_path, index=False)
    y_train.to_csv(train_path.replace('.csv', '_labels.csv'), index=False)
    X_test.to_csv(test_path, index=False)
    y_test.to_csv(test_path.replace('.csv', '_labels.csv'), index=False)

    # Salvar as colunas esperadas pelo modelo
    columns = list(X_encoded.columns)
    with open('columns.json', 'w') as f:
        json.dump(columns, f)

    print(f"Dados pré-processados salvos em {train_path} e {test_path}")
    print("Colunas esperadas salvas em columns.json")

if __name__ == "__main__":
    preprocess_data('../data/raw/datatran2024.csv', '../data/processed/train.csv', '../data/processed/test.csv')
