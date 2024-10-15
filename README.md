# Aprimorando a Segurança no Trânsito: Uma Abordagem de Aprendizado de Máquina para Prever Vítimas Fatais em uma Ocorrência de Acidentes Rodoviários

## Introdução

Os acidentes de trânsito representam uma das principais causas de mortalidade global, configurando um sério problema de saúde pública. Em particular, os acidentes com vítimas fatais têm consequências devastadoras, não apenas para as famílias das vítimas, mas também para a sociedade como um todo, impactando sistemas de saúde, econômicos e sociais. A capacidade de prever com precisão se há vítima fatal numa ocorrência desses acidentes pode ser crucial para o desenvolvimento de políticas públicas eficazes e estratégias preventivas. Modelos preditivos podem auxiliar autoridades a identificar fatores contribuintes, permitindo a alocação direcionada de recursos e implementação de medidas preventivas que potencialmente salvam vidas.

## Descrição e Preparação dos Dados

A base de dados de acidentes da Polícia Rodoviária Federal (PRF) do Brasil é uma coleção abrangente de informações sobre acidentes ocorridos em rodovias federais. Esses dados são disponibilizados ao público em formato aberto, permitindo que pesquisadores, desenvolvedores e o público em geral possam acessar e utilizar as informações para diversos fins, desde a pesquisa acadêmica até o desenvolvimento de soluções para a segurança no trânsito.

Cada registro na base de dados corresponde a uma ocorrência de acidente e contém detalhes sobre o momento, local, condições do acidente, características dos veículos envolvidos, informações sobre as vítimas, etc.

## Metodologia
Para enfrentar o desafio de prever se há ou não vítimas fatais em uma ocorrência de acidente rodoviário, seguimos um processo estruturado de machine learning que abrange desde o pré-processamento de dados até a otimização de modelos. As principais etapas e técnicas utilizadas foram:

### Pré-processamento 
Realizamos a limpeza dos dados, tratamento de valores ausentes e conversão de tipos de dados. 

### Análise Exploratória de Dados (EDA)
A EDA permitiu entender a distribuição das classes, identificar padrões e relações entre variáveis, e detectar possíveis outliers ou inconsistências.
Principais insights incluíram:

- **Distribuição da Classificação dos Acidentes**: Visualizar como os acidentes se distribuem entre diferentes classes forneceu insights iniciais sobre o desequilíbrio presente nos resultados. ![Imagem de exemplo](./imagens/exemplo.png)
- **Análise de Correlação**: Exploramos relações entre características para entender quais fatores contribuem mais para resultados fatais.
![Imagem de exemplo1](./imagens/exemplo1.png)
![Imagem de exemplo2](./imagens/exemplo2.png)

### Engenharia de Recursos (Feature Engineering)

A engenharia de recursos desempenhou um papel crucial em melhorar o poder preditivo de nosso modelo:

- **Recursos Temporais**: Extrair partes do dia como transformar horas em um momento do dia e determinar se um acidente ocorreu em um fim de semana ou não ajudou a capturar padrões temporais.
- **Índice de Severidade de Acidentes**: Combinamos vários campos relacionados a ferimentos para formar um índice composto de severidade de acidentes.
- **Mapear gravidade dos acidentes para valores numéricos**: Poderíamos ter usado os métodos do scikit-learn mas criamos uma coluna 'GRAVIDADE' e mapeamos ``'SEM VÍTIMAS': 0``, ``'COM VÍTIMAS FERIDAS': 1``, ``'COM VÍTIMAS FATAIS': 2'``. Essa é a nossa *variável alvo*.

## Desenvolvimento e Avaliação de Modelos

Experimentamos com vários modelos de aprendizado de máquina e técnicas para prever melhor os resultados da classe com vítimas fatais:

- **Configuração Inicial dos Modelos**: Usando algoritmos de classificação básicos como Regressão Logística, Random Forest e XGBoost, estabelecemos performances de base.
- **Técnicas de Balanceamento de Classes**: Implementamos técnicas como ponderação de classes e geração de dados sintéticos (SMOTE e SMOTEENN) para tratar o desequilíbrio na variável alvo.
**SMOTE (Synthetic Minority Over-sampling Technique):** Técnica de oversampling que gera sinteticamente novas instâncias da classe minoritária ('Com Vítimas Fatais') baseando-se em seus vizinhos mais próximos, equilibrando a distribuição das classes.
**SMOTEENN (SMOTE combinado com Edited Nearest Neighbors):** Combina o SMOTE com o ENN, que realiza undersampling na classe majoritária removendo instâncias próximas à fronteira entre as classes, resultando em um conjunto de dados mais limpo e balanceado.
**Class Weights (Pesos das Classes):** Aplicamos pesos inversamente proporcionais à frequência das classes durante o treinamento dos modelos, penalizando mais os erros cometidos na classe minoritária. Isso força o modelo a prestar mais atenção à classe 'Com Vítimas Fatais'.

- **Ajuste de Hiperparâmetros**: Utilizando `GridSearchCV`, otimizamos os parâmetros do modelo para melhorar o desempenho, focando particularmente no poder preditivo em relação a classe 2.

## Resultados

Os resultados evidenciaram o impacto significativo das técnicas de balanceamento e otimização nas métricas de desempenho para a classe 'Com Vítimas Fatais'. A tabela abaixo resume as métricas de precisão, recall e f1-score para cada modelo e cenário:

### Análise dos Resultados:
- **Melhoria no F1-Score**: Observamos que a combinação de XGBoost com SMOTE + Class Weights atingiu o melhor f1-score para a classe 'Com Vítimas Fatais' (0.510166), tanto com quanto sem pesos nas classes, indicando um bom equilíbrio entre precisão e recall.
- **Impacto das Técnicas de Balanceamento**: O uso de SMOTE e SMOTEENN aumentou o recall em todos os modelos, demonstrando que o balanceamento das classes ajuda os modelos a identificarem mais casos de acidentes com vítimas fatais.
- **Efeito dos Pesos nas Classes**: A aplicação de pesos aumentou o recall, especialmente na Regressão Logística, mas em alguns casos reduziu a precisão, refletindo o trade-off entre identificar mais casos positivos e evitar falsos positivos.
- **Otimização de Hiperparâmetros:** O GridSearchCV contribuiu para melhorias adicionais nos modelos, embora o ganho não tenha sido tão expressivo quanto o proporcionado pelas técnicas de balanceamento.

## Visualizações

Para facilitar a compreensão dos resultados, apresentamos gráficos que comparam as métricas de desempenho entre os modelos e cenários:

- **Gráfico de Barras do F1-Score**: 
- **Gráfico de Linhas do Recall:**: 

*Os gráficos ilustram claramente como cada técnica impactou nas métricas de desempenho, destacando a eficácia das combinações utilizadas.*

## Conclusão

O estudo demonstrou que a aplicação de técnicas de balanceamento de classes e otimização de modelos é essencial para melhorar o desempenho na previsão de acidentes com vítimas fatais. As principais conclusões são:

- **Importância do Balanceamento:** Técnicas como SMOTE e SMOTEENN são eficazes para lidar com desbalanceamento severo, aumentando a capacidade do modelo de detectar a classe minoritária.
- **Trade-off entre Precisão e Recall:** A aplicação de pesos nas classes melhora o recall, mas pode reduzir a precisão. É importante encontrar um equilíbrio adequado dependendo do contexto e dos objetivos do projeto.
- **Otimização de Modelos:** A utilização de GridSearchCV para otimização de hiperparâmetros pode levar a melhorias adicionais, embora seja necessário avaliar o custo computacional envolvido.

## Limitações e Trabalhos Futuros:

- **Qualidade dos Dados:** Algo fundamental seria expandir a nossa base de dados para lidar não somente com ocorrências de 2024 mas também de anos anteriores.
- **Exploração de Novas Técnicas:** Futuras pesquisas podem explorar outras técnicas de balanceamento (e.g., ADASYN, Tomek Links), algoritmos de aprendizado (e.g., Redes Neurais, SVM) e outras estratégias de feature engineering para aprimorar os modelos.
- **Deploy:** O próximo passo seria construir uma API utilizando FASTAPI por exemplo e conteinerizar o projeto para fazer o Deploy. Assim como foi feito neste meu outro projeto aqui: ['github.com/pedarias/MLOps-project-regression'](https://github.com/pedarias/MLOps-project-regression).

## Contribuição

Convidamos feedback, discussões e sugestões para melhoria da comunidade. Sua contribuição é inestimável para refinar nossas abordagens e expandir o impacto desta pesquisa.
