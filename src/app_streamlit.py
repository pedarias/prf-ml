# app_streamlit.py

import streamlit as st
import requests
import json

# Set the page title
st.title('Previsão de Gravidade de Acidentes de Trânsito')

# Load 'columns.json' and parse to get options for categorical variables
with open('columns.json', 'r', encoding='utf-8') as f:
    columns = json.load(f)

# Initialize dictionaries to hold options for each categorical variable
categorical_variables = ['dia_semana', 'uf', 'causa_acidente', 'tipo_acidente',
                         'condicao_metereologica', 'tipo_pista', 'tracado_via',
                         'PERIODO_DIA', 'sentido_via', 'uso_solo', 'fase_dia']

options = {var: [] for var in categorical_variables}

# Parse columns to extract options
for col in columns:
    for var in categorical_variables:
        if col.startswith(var + '_'):
            value = col[len(var) + 1:]  # Remove prefix
            if value not in options[var]:
                options[var].append(value)

# Manually add options for variables not extracted from columns.json
options['PERIODO_DIA'] = ['MADRUGADA', 'MANHÃ', 'NOITE', 'TARDE']
options['sentido_via'] = ['CRESCENTE', 'DECRESCENTE', 'NÃO INFORMADO']
options['uso_solo'] = ['NÃO', 'SIM']
options['fase_dia'] = ['AMANHECER', 'ANOITECER', 'PLENA NOITE', 'PLENO DIA']

# Sort options for better user experience
for var in options:
    options[var] = sorted(options[var])

# Collect user inputs
dia_semana = st.selectbox('Dia da Semana', options['dia_semana'])
uf = st.selectbox('UF', options['uf'])
causa_acidente = st.selectbox('Causa do Acidente', options['causa_acidente'])
tipo_acidente = st.selectbox('Tipo de Acidente', options['tipo_acidente'])
condicao_metereologica = st.selectbox('Condição Meteorológica', options['condicao_metereologica'])
tipo_pista = st.selectbox('Tipo de Pista', options['tipo_pista'])
tracado_via = st.selectbox('Traçado da Via', options['tracado_via'])
PERIODO_DIA = st.selectbox('Período do Dia', options['PERIODO_DIA'])
sentido_via = st.selectbox('Sentido da Via', options['sentido_via'])
uso_solo = st.selectbox('Uso do Solo', options['uso_solo'])
fase_dia = st.selectbox('Fase do Dia', options['fase_dia'])

# Collect numerical inputs
FINAL_DE_SEMANA = st.number_input('Final de Semana (0=Não, 1=Sim)', min_value=0, max_value=1, value=0)
HORA = st.number_input('Hora do Acidente', min_value=0, max_value=23, value=12)
km = st.number_input('KM', value=0.0)
veiculos = st.number_input('Número de Veículos Envolvidos', min_value=1, value=1)
pessoas = st.number_input('Número de Pessoas Envolvidas', min_value=1, value=1)
TOTAL_FERIDOS = st.number_input('Total de Feridos', min_value=0, value=0)

# Prepare the data dictionary
data = {
    'dia_semana': dia_semana.strip().upper(),
    'uf': uf.strip().upper(),
    'causa_acidente': causa_acidente.strip().upper(),
    'tipo_acidente': tipo_acidente.strip().upper(),
    'condicao_metereologica': condicao_metereologica.strip().upper(),
    'tipo_pista': tipo_pista.strip().upper(),
    'tracado_via': tracado_via.strip().upper(),
    'PERIODO_DIA': PERIODO_DIA.strip().upper(),
    'sentido_via': sentido_via.strip().upper(),
    'uso_solo': uso_solo.strip().upper(),
    'fase_dia': fase_dia.strip().upper(),
    'FINAL_DE_SEMANA': int(FINAL_DE_SEMANA),
    'HORA': int(HORA),
    'km': float(km),
    'veiculos': int(veiculos),
    'pessoas': int(pessoas),
    'TOTAL_FERIDOS': int(TOTAL_FERIDOS)
}

# Display the input data (optional, for debugging)
# st.write("Dados enviados para a API:")
# st.json(data)

# Send data to API
if st.button('Obter Previsão'):
    try:
        # Replace with your API URL
        api_url = 'http://localhost:8000/predict'
        response = requests.post(api_url, json=data)
        if response.status_code == 200:
            prediction = response.json()['prediction']
            st.success(f'A previsão é: {prediction[0]}')
        else:
            st.error(f'Erro ao obter a previsão. Status Code: {response.status_code}')
            st.error(f'Resposta da API: {response.text}')
    except Exception as e:
        st.error('Erro ao conectar com a API.')
        st.error(str(e))

