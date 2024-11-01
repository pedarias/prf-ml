import pandas as pd
import numpy as np
import os
import json
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def preprocess_data(input_path, train_path, test_path, pipeline_path):
    """
    Preprocesses the raw data for training a machine learning model.

    Parameters:
    - input_path (str): Path to the raw CSV data file.
    - train_path (str): Path to save the processed training data.
    - test_path (str): Path to save the processed testing data.
    - pipeline_path (str): Path to save the preprocessing pipeline.

    Returns:
    None
    """
    # Read the raw data
    df = pd.read_csv(input_path)
    print(f"Data loaded from {input_path}. Shape: {df.shape}")

    # Drop missing values and duplicates
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    print(f"Data after dropping missing values and duplicates. Shape: {df.shape}")

    # Data type conversion and string cleaning
    df['data_inversa'] = pd.to_datetime(df['data_inversa'])
    df['horario'] = pd.to_datetime(df['horario'], format='%H:%M:%S').dt.time

    # Replace commas with dots and convert to float
    df['km'] = df['km'].astype(str).str.replace(',', '.').astype(float)
    df['latitude'] = df['latitude'].astype(str).str.replace(',', '.').astype(float)
    df['longitude'] = df['longitude'].astype(str).str.replace(',', '.').astype(float)

    # Clean and standardize string columns
    cols_str = ['dia_semana', 'uf', 'municipio', 'causa_acidente', 'tipo_acidente',
                'classificacao_acidente', 'fase_dia', 'sentido_via', 'condicao_metereologica',
                'tipo_pista', 'tracado_via', 'uso_solo', 'regional', 'delegacia', 'uop']
    for col in cols_str:
        df[col] = df[col].astype(str).str.strip().str.upper()

    # Feature engineering
    df['HORA'] = df['horario'].apply(lambda x: x.hour)
    df['PERIODO_DIA'] = df['HORA'].apply(
        lambda x: 'MANHÃ' if 5 <= x < 12 else
                  'TARDE' if 12 <= x < 18 else
                  'NOITE' if 18 <= x < 24 else
                  'MADRUGADA'
    )
    df['FINAL_DE_SEMANA'] = df['dia_semana'].apply(lambda x: 1 if x in ['SÁBADO', 'DOMINGO'] else 0)
    df['TOTAL_FERIDOS'] = df['feridos_leves'] + df['feridos_graves']

    # Map accident severity to numerical values
    gravidade_mapping = {
        'SEM VÍTIMAS': 0,
        'COM VÍTIMAS FERIDAS': 1,
        'COM VÍTIMAS FATAIS': 2
    }
    df['GRAVIDADE'] = df['classificacao_acidente'].map(gravidade_mapping)

    # Feature selection
    features = ['dia_semana', 'uf', 'causa_acidente', 'tipo_acidente',
                'condicao_metereologica', 'tipo_pista', 'tracado_via',
                'PERIODO_DIA', 'sentido_via', 'uso_solo', 'fase_dia',
                'FINAL_DE_SEMANA', 'HORA', 'km', 'veiculos', 'pessoas',
                'TOTAL_FERIDOS']
    X = df[features]
    y = df['GRAVIDADE']
    print(f"Features and target variable selected. X shape: {X.shape}, y shape: {y.shape}")

    # Split the data into training and testing sets before preprocessing
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Data split into training and testing sets.")
    print(f"X_train_raw shape: {X_train_raw.shape}, y_train shape: {y_train.shape}")
    print(f"X_test_raw shape: {X_test_raw.shape}, y_test shape: {y_test.shape}")

    # Define numerical and categorical columns
    numerical_features = ['FINAL_DE_SEMANA', 'HORA', 'km', 'veiculos', 'pessoas', 'TOTAL_FERIDOS']
    categorical_features = ['dia_semana', 'uf', 'causa_acidente', 'tipo_acidente',
                            'condicao_metereologica', 'tipo_pista', 'tracado_via',
                            'PERIODO_DIA', 'sentido_via', 'uso_solo', 'fase_dia']

    # Create preprocessing pipelines for numerical and categorical data
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    # Fit the preprocessor on the training data and transform both training and testing data
    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)
    print("Preprocessing completed.")

    # Get feature names after OneHotEncoding
    ohe = preprocessor.named_transformers_['cat']
    ohe_feature_names = ohe.get_feature_names_out(categorical_features)
    feature_names = numerical_features + list(ohe_feature_names)

    # Convert the arrays back to DataFrames with proper column names
    X_train = pd.DataFrame(X_train, columns=feature_names)
    X_test = pd.DataFrame(X_test, columns=feature_names)

    # Save the processed data
    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    X_train.to_csv(train_path, index=False)
    y_train.to_csv(train_path.replace('.csv', '_labels.csv'), index=False)
    X_test.to_csv(test_path, index=False)
    y_test.to_csv(test_path.replace('.csv', '_labels.csv'), index=False)
    print(f"Processed training data saved to {train_path}")
    print(f"Processed testing data saved to {test_path}")

    # Save the preprocessor for future use during inference
    os.makedirs(os.path.dirname(pipeline_path), exist_ok=True)
    with open(pipeline_path, 'wb') as f:
        pickle.dump(preprocessor, f)
    print(f"Preprocessing pipeline saved to {pipeline_path}")

    # Save the expected columns (feature names) for reference
    with open('columns.json', 'w') as f:
        json.dump(feature_names, f)
    print("Expected feature names saved to columns.json")

    # Optionally, save the raw training and testing data for potential future use
    # Save the raw training and testing data with a tab separator
    X_train_raw.to_csv('../data/processed/X_train_raw.csv', index=False, sep='\t')
    X_test_raw.to_csv('../data/processed/X_test_raw.csv', index=False, sep='\t')
    print("Raw training and testing data saved.")


if __name__ == "__main__":
    preprocess_data(
        input_path='../data/raw/datatran2024.csv',
        train_path='../data/processed/train.csv',
        test_path='../data/processed/test.csv',
        pipeline_path='../artifacts/preprocessor.pkl'
    )
