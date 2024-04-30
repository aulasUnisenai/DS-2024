# -*- coding: utf-8 -*-
!pip install kneed

import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler)
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from kneed import KneeLocator
import numpy as np

"""# Dados"""

# Carregar base
df = pd.read_csv('/content/marketing.csv', sep= '\t')
df.head(5)

# Informações
df.info()

# Verificar NaN
df.isna().sum()

# Preencher valores vazios com a mediana
df['renda']= df['renda'].fillna(df['renda'].median())

# Verificar duplicados
df[df.duplicated()]

# Verificar valores únicos
df.nunique()

# Remover colunas
cols = ['ID', 'custoContatoZ', 'receitaZ']

for col in cols:
  df.drop(col, axis =1, inplace =True)

# Alterar dados
cols = ['anoNascimento','aceitouCampanha3',
        'aceitouCampanha4', 'aceitouCampanha2',
        'aceitouCampanha1','aceitouCampanha5',
        'reclamacao', 'resposta'
        ]

df[cols] = df[cols].astype('object')

"""# Engenharia de Variáveis"""

# Idade
df["idade"] = 2024-df["anoNascimento"]
df["idade"]

# Convertendo a coluna de datas para o tipo datetime
df['dataCliente'] = pd.to_datetime(df['dataCliente'],
                                   format='%d-%m-%Y')

# Tempo cliente
df["tempoCliente"] = 2024-df["dataCliente"].dt.year
df["tempoCliente"]

# Total gasto
df["totalGasto"] = (df["vinhos"] +
                    df["frutas"] +
                    df["carne"] +
                    df["peixe"] +
                    df["doces"] +
                    df["ouro"])
df["totalGasto"]

# Mora sozinho
df["moraSozinho"] = df["estadoCivil"].replace({
    "Married":"nao",
    "Together":"nao",
    "Absurd":"sim",
    "Widow":"sim",
    "YOLO":"sim",
    "Alone": "sim",
    "Divorced":"sim",
    "Single":"sim",})

df["moraSozinho"]

# Mora com crianças/adolescentes
df["criancas"]= df["filhosCasa"]+df["adolescentesCasa"]
df["criancas"]

# Tamanho da família
df["tamanhoFamilia"] = df["moraSozinho"].replace(
    {"sim": 1, "nao":2}) + df["criancas"]

df["tamanhoFamilia"]

# Grupos educacionas
df["educacao"] = df["educacao"].replace(
    {"Basic":"naoGraduado",
     "2n Cycle":"naoGraduado",
     "Graduation":"graduado",
     "Master":"posGraduado",
     "PhD":"posGraduado"})

df["educacao"]

# Alterar dados
df['moraSozinho'] = df['moraSozinho'].astype('object')
df["idade"] = df["idade"].astype('int')

# Remover colunas redundantes
cols = ["anoNascimento", "estadoCivil", "dataCliente"]

for col in cols:
  df.drop(col, axis =1, inplace =True)

# Colunas numéricas
colunas_numericas = df.select_dtypes(
                include=['int', 'float']).columns
colunas_numericas

# Colunas categóricas
colunas_categoricas = df.select_dtypes(
                 include=['object']).columns
colunas_categoricas

# Valores Únicos categóricos
for col in colunas_categoricas:
    print(col, df[col].unique())

"""# EDA"""

# Estatísticas descritivas
df.describe().T.round(2)

# Correlações
corr = df.corr(numeric_only=True)
corr.style.background_gradient(cmap='coolwarm')

# Histograma e boxplot
for col in colunas_numericas:
   fig = px.histogram(df,
                      x=col, marginal="box",
                      template='plotly_dark')
   fig.show()

# Gráfico de barras
for col in colunas_categoricas:
    counts = df[col].value_counts()

    # Criar gráfico de barras
    fig = px.bar(x=counts.index,
                 y=counts.values,
                 text_auto = True,
                 template='plotly_dark')

    fig.update_traces(textposition='outside')

    fig.update_layout(xaxis_title=col,
                      yaxis_title='Contagem',
                      )

    fig.show()

# Remover outliers
df = df[(df["idade"]<90)]
df = df[(df["renda"]<600000)]

"""# Pré-processamento"""

# Transformador para colunas numéricas
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# Transformador para colunas numéricas
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder())
])

preprocessor = ColumnTransformer(
  transformers=[
    ('num', numeric_transformer, colunas_numericas),
    ('cat', categorical_transformer, colunas_categoricas)
    ])

# Criando o pipeline com o pré-processamento e o KMeans
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('kmeans', KMeans(n_init='auto'))])

# Calculando a inércia para diferentes valores de k
inercia = []

for k in range(2, 6):
    pipeline.set_params(kmeans__n_clusters=k)  # Definindo o número de clusters
    pipeline.fit(df)
    # Obtendo os labels do cluster antes da codificação one-hot
    labels = pipeline.named_steps['kmeans'].labels_
    inercia.append(pipeline.named_steps['kmeans'].inertia_)

# Encontrar o número ideal de clusters (ponto de cotovelo)
diff = np.diff(inercia)  # Calcula as diferenças entre os valores de inércia
knee_index = np.argmax(diff) + 1  # Encontra o índice do ponto de cotovelo
ideal_num_clusters = knee_index + 2  # Adiciona 2 devido ao range começando em 2

# Plotando o método do cotovelo com a linha para indicar o número ideal de clusters
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(range(2, 6), inercia, marker='o')
plt.axvline(x=ideal_num_clusters, color='r',
            linestyle='--', label='Número ideal de clusters')
plt.xlabel('Número de clusters')
plt.ylabel('Inércia')
plt.title('Método do Cotovelo')
plt.legend()


plt.tight_layout()
plt.show()

"""# Agrupamento"""

# Aplicando KMeans após pré-processamento
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('kmeans', KMeans(n_clusters=4))])

# Fit do modelo
pipeline.fit(df)

# Obtendo os clusters
clusters = pipeline.predict(df)


# Adicionando os clusters ao DataFrame
df['cluster'] = clusters
df.head(4)

"""# Análises"""

# Verificar a distribuição dos grupos
counts = df['cluster'].value_counts()
counts

# Gráfico de dispersão colorido por cluster usando Plotly
fig = px.scatter(df, x='totalGasto',
                 y='renda',
                 color='cluster',
                 title='Gasto por Grupo',
                 labels={'totalGasto': 'Gasto Total',
                         'renda': 'Renda'})

# Exibindo o gráfico
fig.show()

# Criando o gráfico de caixa e bigodes com enxame de pontos
fig = px.box(df, x='cluster',
             y='totalGasto',
             points='all',
             title='Gasto por Grupo')

# Exibindo o gráfico
fig.update_traces(marker=dict(color='red',
                              opacity=0.5),
                  jitter=0.3)
fig.show()

# Calculando estatísticas descritivas para cada cluster
cluster_describe = (df.groupby('cluster')
[colunas_numericas].describe())

# Exibindo as estatísticas descritivas por cluster
print(cluster_describe.round(2).T.to_string())
