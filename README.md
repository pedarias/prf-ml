# Aprimorando a Segurança no Trânsito: Uma Abordagem de Aprendizado de Máquina para Prever Vítimas Fatais em uma Ocorrência de Acidentes Rodoviários

## Introdução

Os acidentes de trânsito representam uma das principais causas de mortalidade global, configurando um sério problema de saúde pública. Em particular, os acidentes com vítimas fatais têm consequências devastadoras, não apenas para as famílias das vítimas, mas também para a sociedade como um todo, impactando sistemas de saúde, econômicos e sociais. A capacidade de prever com precisão se há vítima fatal numa ocorrência desses acidentes pode ser crucial para o desenvolvimento de políticas públicas eficazes e estratégias preventivas. Modelos preditivos podem auxiliar autoridades a identificar *fatores contribuintes*, permitindo a alocação direcionada de recursos e implementação de medidas preventivas que potencialmente salvam vidas.

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

Experimentamos com vários modelos de aprendizado de máquina e técnicas para prever melhor os resultados da classe com vítimas fatais. 
O tratamento de pesos de classe e técnicas de amostragem é realizado de maneira cuidadosa e adaptada a cada cenário específico de treinamento. 
### Configuração de Pesos de Classe e Amostragem
O script configura inicialmente os pesos de classe antes de entrar nos loops de treinamento para cada modelo e configuração de amostragem. Esses pesos são calculados para a distribuição original de classes (ou seja, antes da aplicação de qualquer técnica de amostragem como SMOTE ou SMOTEENN). Isso é feito usando compute_class_weight da ``sklearn.utils.class_weight`` para gerar um dicionário de pesos que será aplicado aos modelos que suportam o parâmetro class_weight diretamente.
Pesos inversamente proporcionais à frequência das classes, penalizando mais os erros cometidos na classe minoritária.

### Tratamento de Dados e Amostragem
Para cada modelo, aplicamos diferentes configurações de amostragem (None, SMOTE, SMOTEENN) para entender como cada técnica afeta o desempenho do modelo. Vamos detalhar cada cenário:

- **Cenário "None"**Os pesos são calculados para a distribuição original de classes do conjunto de dados e utilizados diretamente nos modelos. Não é aplicada nenhuma técnica de amostragem. Os dados de treino são usados como estão.

- **Cenário "SMOTE"**O SMOTE é aplicado para reamostrar o conjunto de dados de treino, equilibrando as classes aumentando a quantidade de amostras da classe minoritária. Depois de aplicar o SMOTE, os pesos de classe são recalculados com base na nova distribuição de classes do conjunto reamostrado. Isso é importante porque a distribuição das classes mudou, o que pode alterar a importância relativa das classes no treinamento do modelo.

- **Cenário "SMOTEENN"**Amostragem SMOTEENN combina o SMOTE (Synthetic Minority Over-sampling Technique) com o ENN (Edited Nearest Neighbors), uma técnica de limpeza que pode remover amostras de ambas as classes que são consideradas como ruído. Assim como com o SMOTE, os pesos de classe são recalculados para refletir as mudanças na distribuição das classes após a aplicação do SMOTEENN.


## Processo de Treinamento
Dentro de cada iteração do modelo e configuração de amostragem:

- **Definição do Modelo:** O modelo é (re)definido com os parâmetros atualizados, incluindo os novos pesos de classe, se aplicável.

- **Treinamento:** O modelo é treinado no conjunto de dados de treino (original, SMOTE ou SMOTEENN), utilizando os pesos recalculados, se aplicável.

- **Avaliação:** O modelo é avaliado no conjunto de teste para calcular métricas como acurácia, relatório de classificação e matriz de confusão.

- **Registro no MLflow:** Cada configuração (modelo + técnica de amostragem) é registrada como uma execução separada no MLflow, onde parâmetros, métricas e artefatos (por exemplo, matrizes de confusão) são armazenados.

- **Ajuste de Hiperparâmetros**: Na amostragem com SMOTEENN, utilizamos `RandomizedSearchCV` a fim de otimizarmos os parâmetros do modelo para melhorar o desempenho, focando particularmente no poder preditivo em relação a classe 2('Com vitimas fatais').

## Resultados

Os resultados evidenciaram o impacto significativo das técnicas de balanceamento e otimização nas métricas de desempenho para a classe 'Com Vítimas Fatais'. A tabela abaixo resume as métricas de precisão, recall e f1-score para cada modelo e cenário, somente para a classe ``'Com Vitimas Fatais'``:

| Modelo                     | Cenário                                 | Precisão | Recall | F1-Score |
|-----------------------------|--------------------------------------   |----------|--------|----------|
| Logistic Regression          | Class Weights                           | 0.257    | 0.636  | 0.366    |
| Logistic Regression          | SMOTE + Class Weights                   | 0.511    | 0.272  | 0.355    |
| Logistic Regression          | SMOTEENN + Class Weights                | 0.341    | 0.389  | 0.364    |
| Logistic Regression          | SMOTEENN + Class Weights + GridSearchCV | 0.286    | 0.476  | 0.357    |
| Random Forest                | Class Weights                           | 0.830    | 0.280  | 0.419    |
| Random Forest                | SMOTE + Class Weights                   | 0.759    | 0.342  | 0.472    |
| Random Forest                | SMOTEENN + Class Weights                | 0.458    | 0.483  | 0.470    |
| Random Forest                | SMOTEENN + Class Weights + GridSearchCV | 0.454    | 0.474  | 0.464    |
| XGBoost                      | Class Weights                           | 0.423    | 0.590  | 0.493    |
| XGBoost                      | SMOTE + Class Weights                   | 0.688    | 0.405  | 0.510    |
| XGBoost                      | SMOTEENN + Class Weights                | 0.494    | 0.507  | 0.500    |
| XGBoost                      | SMOTEENN + Class Weights + GridSearchCV | 0.452    | 0.545  | 0.494    |


### Análise dos Resultados:
#### Trade-Off entre Classes
As técnicas de balanceamento e otimização impactam diretamente no equilíbrio entre Precisão e Recall, criando um trade-off que precisa ser cuidadosamente gerenciado:
- **Precisão:** Alta Precisão indica que poucas instâncias preditas como 'Com Vítimas Fatais' são falsas positivas. No entanto, focar excessivamente na Precisão pode resultar em baixo Recall, ou seja, o modelo deixa de identificar muitos casos reais dessa classe.
- **Recall:** Alto Recall significa que o modelo consegue identificar a maioria das instâncias reais de 'Com Vítimas Fatais'. Contudo, isso pode acarretar em baixa Precisão, aumentando o número de falsos positivos.

A aplicação de Class Weights inicialmente aumentou o Recall para todas as classes, mas resultou em variações na Precisão. A introdução de SMOTE melhorou a Precisão e moderadamente a Recall para alguns modelos, enquanto SMOTEENN ofereceu um equilíbrio melhorado entre ambas as métricas. A otimização com GridSearchCV permitiu ajustar os hiperparâmetros para maximizar o F1-Score, que é uma métrica que combina Precisão e Recall, oferecendo uma visão mais equilibrada do desempenho do modelo.


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

As técnicas de balanceamento aplicadas tiveram efeitos significativos nas métricas de desempenho para a classe 'Com Vítimas Fatais'. Enquanto a aplicação de Class Weights inicialmente melhorou o Recall, técnicas como SMOTE e SMOTEENN ajudaram a equilibrar as métricas, aumentando tanto a Precisão quanto o Recall em diferentes graus. A otimização de hiperparâmetros com GridSearchCV permitiu ajustes finos, melhorando o F1-Score em alguns casos, embora a Precisão e o Recall nem sempre tenham aumentado simultaneamente.

Este trade-off entre Precisão e Recall é intrínseco ao lidar com classes desbalanceadas. A escolha da técnica de balanceamento e a otimização dos parâmetros do modelo devem ser guiadas pelos objetivos específicos do projeto. No contexto da previsão de acidentes fatais, priorizar o Recall pode ser mais adequado para garantir que a maioria dos casos críticos sejam identificados, mesmo que isso implique em um aumento no número de falsos positivos.
## Limitações e Trabalhos Futuros:

- **Qualidade dos Dados:** Algo fundamental seria expandir a nossa base de dados para lidar não somente com ocorrências de 2024 mas também de anos anteriores.
- **Exploração de Outras Técnicas de Balanceamento:** Futuras pesquisas podem explorar outras técnicas de balanceamento (e.g., ADASYN, Tomek Links, Cluster-Based Over Sampling), algoritmos de aprendizado (e.g., SVM, Redes Neurais) e outras estratégias de feature engineering para aprimorar os modelos.
- **Deploy:** O próximo passo seria construir uma API utilizando FastAPI por exemplo e conteinerizar o projeto para fazer o Deploy. Assim como foi feito neste meu outro projeto [aqui](https://github.com/pedarias/MLOps-project-regression).

## Contribuição

Convidamos feedback, discussões e sugestões para melhoria da comunidade. Sua contribuição é inestimável para refinar nossas abordagens e expandir o impacto desta pesquisa.

