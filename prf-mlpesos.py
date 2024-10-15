# 1. Importação de Bibliotecas e Configurações Iniciais

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configurações de plot
sns.set_style('darkgrid')
sns.set_theme(rc={'figure.figsize': (12, 6)})

# Importações adicionais para MLflow
import mlflow
import mlflow.sklearn
import mlflow.xgboost

# 2. Carregamento e Pré-processamento dos Dados

from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

# Importação adicional para calcular os pesos das classes
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight

# Carregamento dos dados
df = pd.read_csv('datatran2024.csv', sep=';', encoding='latin1')

# Verificação inicial
print(df.info())
print(df.head())

# 2.1. Tratamento de Valores Ausentes e Dados Duplicados

# Remover linhas com valores ausentes
df.dropna(inplace=True)

# Verificar duplicatas
df.drop_duplicates(inplace=True)

# Verificação após limpeza
print(df.info())

# 2.2. Conversão de Tipos de Dados

# Converter 'data_inversa' para datetime
df['data_inversa'] = pd.to_datetime(df['data_inversa'])

# Converter 'horario' para datetime.time
df['horario'] = pd.to_datetime(df['horario'], format='%H:%M:%S').dt.time

# Converter 'km' para float
df['km'] = df['km'].astype(str).str.replace(',', '.').astype(float)

# Converter 'latitude' e 'longitude' para float
df['latitude'] = df['latitude'].astype(str).str.replace(',', '.').astype(float)
df['longitude'] = df['longitude'].astype(str).str.replace(',', '.').astype(float)

# 2.3. Padronização de Strings

# Remover espaços extras e padronizar strings
cols_str = ['dia_semana', 'uf', 'municipio', 'causa_acidente', 'tipo_acidente',
            'classificacao_acidente', 'fase_dia', 'sentido_via', 'condicao_metereologica',
            'tipo_pista', 'tracado_via', 'uso_solo', 'regional', 'delegacia', 'uop']

for col in cols_str:
    df[col] = df[col].astype(str).str.strip().str.upper()

# 3. Análise Exploratória de Dados (EDA)
# Você pode incluir suas análises aqui, lembrando de adicionar plt.show() após cada plot.

# Exemplo:
# Distribuição da Classificação dos Acidentes
class_counts = df['classificacao_acidente'].value_counts()
sns.countplot(x='classificacao_acidente', data=df, order=class_counts.index)
plt.title('Distribuição da Classificação dos Acidentes')
plt.xlabel('Classificação do Acidente')
plt.ylabel('Número de Acidentes')
plt.show()

# Continue com as demais análises...

# 4. Feature Engineering

# Criar coluna 'HORA'
df['HORA'] = df['horario'].apply(lambda x: x.hour)

# Período do Dia
def get_periodo_dia(hora):
    if 5 <= hora < 12:
        return 'MANHÃ'
    elif 12 <= hora < 18:
        return 'TARDE'
    elif 18 <= hora < 24:
        return 'NOITE'
    else:
        return 'MADRUGADA'

df['PERIODO_DIA'] = df['HORA'].apply(get_periodo_dia)

# Final de Semana
df['FINAL_DE_SEMANA'] = df['dia_semana'].apply(lambda x: 1 if x in ['SÁBADO', 'DOMINGO'] else 0)

# Quantidade Total de Feridos
df['TOTAL_FERIDOS'] = df['feridos_leves'] + df['feridos_graves']

# Gravidade do Acidente (mapear para valores numéricos)
gravidade_mapping = {
    'SEM VÍTIMAS': 0,
    'COM VÍTIMAS FERIDAS': 1,
    'COM VÍTIMAS FATAIS': 2
}
df['GRAVIDADE'] = df['classificacao_acidente'].map(gravidade_mapping)

# 5. Preparação dos Dados para o Modelo

# Selecionar features e target
features = ['dia_semana', 'uf', 'causa_acidente', 'tipo_acidente',
            'condicao_metereologica', 'tipo_pista', 'tracado_via',
            'PERIODO_DIA', 'sentido_via', 'uso_solo', 'fase_dia',
            'FINAL_DE_SEMANA', 'HORA', 'km', 'veiculos', 'pessoas',
            'TOTAL_FERIDOS']

X = df[features]
y = df['GRAVIDADE']

# Identificar colunas categóricas e numéricas
cat_cols = X.select_dtypes(include=['object']).columns.tolist()
num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# One-Hot Encoding
X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=True)

# Padronização das variáveis numéricas
scaler = StandardScaler()
X_encoded[num_cols] = scaler.fit_transform(X_encoded[num_cols])

# Divisão em conjunto de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y)

# 5.1. Cálculo dos Pesos das Classes

# Calcular os pesos das classes
class_weights = compute_class_weight(
    class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(zip(np.unique(y_train), class_weights))

print("Class Weights:", class_weights)

# Calcular os sample weights para XGBoost
sample_weights = compute_sample_weight(
    class_weight='balanced', y=y_train)

# 6. Definição de Funções Auxiliares

from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN

# Função para treinar e avaliar modelos
def train_evaluate_model(model, X_train, y_train, X_test, y_test, model_name, sample_weights=None):
    # Treinar o modelo
    if sample_weights is not None:
        model.fit(X_train, y_train, sample_weight=sample_weights)
    else:
        model.fit(X_train, y_train)

    # Prever no conjunto de teste
    y_pred = model.predict(X_test)

    # Avaliar o modelo
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred, target_names=['Sem Vítimas', 'Com Vítimas Feridas', 'Com Vítimas Fatais'], output_dict=True)

    # Exibir resultados
    print(f"Modelo: {model_name}")
    print("Accuracy:", accuracy)
    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred, target_names=['Sem Vítimas', 'Com Vítimas Feridas', 'Com Vítimas Fatais']))

    # Matriz de Confusão
    cm = confusion_matrix(y_test, y_pred)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Sem Vítimas', 'Com Vítimas Feridas', 'Com Vítimas Fatais'],
                yticklabels=['Sem Vítimas', 'Com Vítimas Feridas', 'Com Vítimas Fatais'])
    plt.xlabel('Predito')
    plt.ylabel('Verdadeiro')
    plt.title(f'Matriz de Confusão - {model_name}')
    plt.savefig(f"confusion_matrix_{model_name}.png")
    plt.close()

    return accuracy, report

# Função para aplicar técnicas de balanceamento
def apply_sampling(sampling_technique, X_train, y_train):
    if sampling_technique == 'SMOTE':
        sampler = SMOTE(random_state=42)
    elif sampling_technique == 'SMOTEENN':
        sampler = SMOTEENN(random_state=42)
    else:
        return X_train, y_train  # Sem balanceamento

    X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
    return X_resampled, y_resampled

# Função para realizar GridSearchCV
def perform_grid_search(model, param_grid, X, y, sample_weights=None, scoring='f1_macro'):
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=3,
        scoring=scoring,
        n_jobs=-1
    )
    if sample_weights is not None:
        grid_search.fit(X, y, sample_weight=sample_weights)
    else:
        grid_search.fit(X, y)
    best_model = grid_search.best_estimator_
    print(f"Melhores hiperparâmetros para {model.__class__.__name__}:")
    print(grid_search.best_params_)
    return best_model

# 7. Avaliação dos Modelos em Diferentes Cenários

# 7.1. Definição dos Modelos

# Modelos a serem avaliados
models = {
    'Random Forest': RandomForestClassifier(class_weight=class_weights, random_state=42),
    'XGBoost': xgb.XGBClassifier(eval_metric='mlogloss', random_state=42),
    'Logistic Regression': LogisticRegression(class_weight=class_weights, max_iter=5000, random_state=42)
}

# 7.2. Avaliação com Class Weights

# Dicionário para armazenar os resultados
results = {}

# Iniciar o MLflow
mlflow.set_experiment("Acidentes de Trânsito")

for model_name, model in models.items():
    with mlflow.start_run(run_name=f"{model_name} com Class Weights"):
        if model_name == 'XGBoost':
            print(f"\nAvaliação do {model_name} com Class Weights:")
            accuracy, report = train_evaluate_model(
                model, X_train, y_train, X_test, y_test, model_name + " com Class Weights", sample_weights=sample_weights)
        else:
            print(f"\nAvaliação do {model_name} com Class Weights:")
            accuracy, report = train_evaluate_model(
                model, X_train, y_train, X_test, y_test, model_name + " com Class Weights")
        results[(model_name, 'Class Weights')] = report

        # Logar parâmetros e métricas no MLflow
        mlflow.log_param("sampling", "None")
        mlflow.log_param("class_weight", "Balanced")
        mlflow.log_param("model_name", model_name)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score_fatal", report['Com Vítimas Fatais']['f1-score'])
        mlflow.log_metric("recall_fatal", report['Com Vítimas Fatais']['recall'])
        mlflow.log_metric("precision_fatal", report['Com Vítimas Fatais']['precision'])

        # Salvar a matriz de confusão como artefato
        mlflow.log_artifact(f"confusion_matrix_{model_name} com Class Weights.png")

        # Salvar o modelo
        if model_name == 'XGBoost':
            mlflow.xgboost.log_model(model, artifact_path="model")
        else:
            mlflow.sklearn.log_model(model, artifact_path="model")

# 7.3. Avaliação com SMOTE e Class Weights

# Aplicar SMOTE
X_train_smote, y_train_smote = apply_sampling('SMOTE', X_train, y_train)

# Recalcular os pesos das classes para o novo conjunto
class_weights_smote = compute_class_weight(
    class_weight='balanced', classes=np.unique(y_train_smote), y=y_train_smote)
class_weights_smote = dict(zip(np.unique(y_train_smote), class_weights_smote))
sample_weights_smote = compute_sample_weight(
    class_weight='balanced', y=y_train_smote)

for model_name, model in models.items():
    with mlflow.start_run(run_name=f"{model_name} com SMOTE e Class Weights"):
        if model_name == 'XGBoost':
            print(f"\nAvaliação do {model_name} com SMOTE e Class Weights:")
            # Atualizar o modelo com os novos sample weights
            model = xgb.XGBClassifier(eval_metric='mlogloss', random_state=42)
            accuracy, report = train_evaluate_model(
                model, X_train_smote, y_train_smote, X_test, y_test, model_name + " com SMOTE e Class Weights", sample_weights=sample_weights_smote)
        else:
            print(f"\nAvaliação do {model_name} com SMOTE e Class Weights:")
            # Atualizar o modelo com os novos class weights
            model.set_params(class_weight=class_weights_smote)
            accuracy, report = train_evaluate_model(
                model, X_train_smote, y_train_smote, X_test, y_test, model_name + " com SMOTE e Class Weights")
        results[(model_name, 'SMOTE + Class Weights')] = report

        # Logar parâmetros e métricas no MLflow
        mlflow.log_param("sampling", "SMOTE")
        mlflow.log_param("class_weight", "Balanced")
        mlflow.log_param("model_name", model_name)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score_fatal", report['Com Vítimas Fatais']['f1-score'])
        mlflow.log_metric("recall_fatal", report['Com Vítimas Fatais']['recall'])
        mlflow.log_metric("precision_fatal", report['Com Vítimas Fatais']['precision'])

        # Salvar a matriz de confusão como artefato
        mlflow.log_artifact(f"confusion_matrix_{model_name} com SMOTE e Class Weights.png")

        # Salvar o modelo
        if model_name == 'XGBoost':
            mlflow.xgboost.log_model(model, artifact_path="model")
        else:
            mlflow.sklearn.log_model(model, artifact_path="model")

# 7.4. Avaliação com SMOTEENN e Class Weights

# Aplicar SMOTEENN
X_train_smoteenn, y_train_smoteenn = apply_sampling('SMOTEENN', X_train, y_train)

# Recalcular os pesos das classes para o novo conjunto
class_weights_smoteenn = compute_class_weight(
    class_weight='balanced', classes=np.unique(y_train_smoteenn), y=y_train_smoteenn)
class_weights_smoteenn = dict(zip(np.unique(y_train_smoteenn), class_weights_smoteenn))
sample_weights_smoteenn = compute_sample_weight(
    class_weight='balanced', y=y_train_smoteenn)

for model_name, model in models.items():
    with mlflow.start_run(run_name=f"{model_name} com SMOTEENN e Class Weights"):
        if model_name == 'XGBoost':
            print(f"\nAvaliação do {model_name} com SMOTEENN e Class Weights:")
            # Atualizar o modelo com os novos sample weights
            model = xgb.XGBClassifier(eval_metric='mlogloss', random_state=42)
            accuracy, report = train_evaluate_model(
                model, X_train_smoteenn, y_train_smoteenn, X_test, y_test, model_name + " com SMOTEENN e Class Weights", sample_weights=sample_weights_smoteenn)
        else:
            print(f"\nAvaliação do {model_name} com SMOTEENN e Class Weights:")
            # Atualizar o modelo com os novos class weights
            model.set_params(class_weight=class_weights_smoteenn)
            accuracy, report = train_evaluate_model(
                model, X_train_smoteenn, y_train_smoteenn, X_test, y_test, model_name + " com SMOTEENN e Class Weights")
        results[(model_name, 'SMOTEENN + Class Weights')] = report

        # Logar parâmetros e métricas no MLflow
        mlflow.log_param("sampling", "SMOTEENN")
        mlflow.log_param("class_weight", "Balanced")
        mlflow.log_param("model_name", model_name)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score_fatal", report['Com Vítimas Fatais']['f1-score'])
        mlflow.log_metric("recall_fatal", report['Com Vítimas Fatais']['recall'])
        mlflow.log_metric("precision_fatal", report['Com Vítimas Fatais']['precision'])

        # Salvar a matriz de confusão como artefato
        mlflow.log_artifact(f"confusion_matrix_{model_name} com SMOTEENN e Class Weights.png")

        # Salvar o modelo
        if model_name == 'XGBoost':
            mlflow.xgboost.log_model(model, artifact_path="model")
        else:
            mlflow.sklearn.log_model(model, artifact_path="model")

# 7.5. Avaliação com SMOTEENN, Class Weights e GridSearchCV

# Parâmetros para GridSearchCV
param_grids = {
    'Random Forest': {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'class_weight': [class_weights_smoteenn, 'balanced', None]
    },
    'XGBoost': {
        'n_estimators': [100, 200],
        'max_depth': [3, 6, 10],
        'learning_rate': [0.01, 0.1, 0.2]
    
    },
    'Logistic Regression': {
        'penalty': ['l2'],
        'C': [0.1, 1.0, 10.0],
        'class_weight': [class_weights_smoteenn, 'balanced', None],
        'solver': ['lbfgs', 'saga']
    }
}

for model_name, model in models.items():
    with mlflow.start_run(run_name=f"{model_name} com SMOTEENN + Class Weights + GridSearchCV"):
        print(f"\nAvaliação do {model_name} com SMOTEENN, Class Weights e GridSearchCV:")
        param_grid = param_grids[model_name]
        if model_name == 'XGBoost':
            best_model = perform_grid_search(
                model, param_grid, X_train_smoteenn, y_train_smoteenn, sample_weights=sample_weights_smoteenn)
            accuracy, report = train_evaluate_model(
                best_model, X_train_smoteenn, y_train_smoteenn, X_test, y_test,
                model_name + " Otimizado com SMOTEENN e Class Weights", sample_weights=sample_weights_smoteenn)
        else:
            best_model = perform_grid_search(
                model, param_grid, X_train_smoteenn, y_train_smoteenn)
            accuracy, report = train_evaluate_model(
                best_model, X_train_smoteenn, y_train_smoteenn, X_test, y_test,
                model_name + " Otimizado com SMOTEENN e Class Weights")

        results[(model_name, 'SMOTEENN + Class Weights + GridSearchCV')] = report

        # Logar parâmetros e métricas no MLflow
        mlflow.log_param("sampling", "SMOTEENN")
        mlflow.log_param("model_name", model_name)

        # Preparar os parâmetros para registro
        params = best_model.get_params()

        # Converter 'class_weight' se necessário
        if 'class_weight' in params:
            if isinstance(params['class_weight'], dict):
                # Converter chaves e valores para tipos básicos do Python
                params['class_weight'] = {int(k): float(v) for k, v in params['class_weight'].items()}
            elif params['class_weight'] is None:
                params['class_weight'] = 'None'
            else:
                params['class_weight'] = str(params['class_weight'])

        # Certificar-se de que todos os valores são serializáveis
        for key in params:
            value = params[key]
            if isinstance(value, (np.integer, np.floating)):
                params[key] = value.item()
            elif not isinstance(value, (int, float, str, bool)):
                params[key] = str(value)

        mlflow.log_params(params)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score_fatal", report['Com Vítimas Fatais']['f1-score'])
        mlflow.log_metric("recall_fatal", report['Com Vítimas Fatais']['recall'])
        mlflow.log_metric("precision_fatal", report['Com Vítimas Fatais']['precision'])

        # Salvar a matriz de confusão como artefato
        mlflow.log_artifact(f"confusion_matrix_{model_name} Otimizado com SMOTEENN e Class Weights.png")

        # Salvar o modelo
        if model_name == 'XGBoost':
            mlflow.xgboost.log_model(best_model, artifact_path="model")
        else:
            mlflow.sklearn.log_model(best_model, artifact_path="model")


# 8. Comparação dos Resultados em uma Tabela

# Criar uma tabela consolidada com as principais métricas
metrics = ['precision', 'recall', 'f1-score']
classes = ['Sem Vítimas', 'Com Vítimas Feridas', 'Com Vítimas Fatais']

rows = []
for (model_name, scenario), report in results.items():
    for cls in classes:
        row = {
            'Modelo': model_name,
            'Cenário': scenario,
            'Classe': cls
        }
        for metric in metrics:
            row[metric] = report[cls][metric]
        rows.append(row)

results_df = pd.DataFrame(rows)

# Pivotar a tabela para melhor visualização
pivot_df = results_df.pivot_table(index=['Modelo', 'Cenário'], columns='Classe', values=metrics)
print(pivot_df)

# 9. Conclusões

# *Aqui você pode analisar os resultados e tirar conclusões sobre qual modelo e técnica de balanceamento apresentou o melhor desempenho, especialmente na classe "Com Vítimas Fatais".*
