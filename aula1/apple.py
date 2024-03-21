# -*- coding: utf-8 -*-
"""apple.ipynb

# Base
"""

import pandas as pd

# Carregar a base de dados
df = pd.read_csv('/content/apple.csv')
df.head(5)

# Informações sobre a base
df.info()

# Alterar o tipo do dado
df['acidez'] = pd.to_numeric(df['acidez'], errors='coerce')

# Verificar duplicados
df.duplicated().sum()

# Verificar NA
df.isna().sum()

# Remover NA
df = df.dropna()
df.isna().sum()

"""# EDA"""

import plotly.express as px
import plotly.figure_factory as ff

# Excluir a coluna 'id'
df.drop('id', axis=1, inplace=True)

df.describe().T.round(2)

# Selecionar colunas numéricas
df_numericas = df.select_dtypes(include=['int', 'float']).columns
df_numericas

# Criar histogramas para cada variável numérica
for col in df_numericas:
    fig = px.histogram(df, x=col, color = df.qualidade)
    fig.show()

# Correlação
corr_matrix = df.corr(numeric_only=True).round(2)
fig = ff.create_annotated_heatmap(z=corr_matrix.values,
                                  x=list(corr_matrix.columns),
                                  y=list(corr_matrix.index),
                                  colorscale='Viridis')
fig.update_layout(title='Matriz de Correlação')
fig.show()

# Criar tabela de contagem para "qualidade"
contagem_qualidade = df['qualidade'].value_counts().reset_index()
contagem_qualidade.columns = ['Qualidade', 'Contagem']

# Criar gráfico de barras personalizado
fig = px.bar(contagem_qualidade,
             x='Qualidade', y='Contagem',
             title='Distribuição da Qualidade',
             color='Qualidade',
             color_discrete_map={'boa': 'blue', 'ruim': 'orange'},
             width=700,  # Largura das barras
             height=500  # Altura do gráfico
            )

fig.update_layout(xaxis_title='Qualidade', yaxis_title='Contagem')
fig.show()

"""# Classificação - Simples"""

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

# Definir as features (X) e a variável alvo (y)
X = df.drop(['qualidade'], axis=1)
y = df['qualidade']

# Dividir os dados em conjunto de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criar pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier())
])

# Treinar o modelo no conjunto de treino
pipeline.fit(X_train, y_train)

# Realizar previsões no conjunto de teste
y_pred = pipeline.predict(X_test)

# Avaliar o desempenho do modelo
acuracia = accuracy_score(y_test, y_pred)
relatorio_classificacao = classification_report(y_test, y_pred)

print('\nRelatório de Classificação:\n', relatorio_classificacao)

# Criar a matriz de confusão
conf_matrix = confusion_matrix(y_test, y_pred, labels=['boa', 'ruim'])

# Criar um heatmap interativo da matriz de confusão
fig = ff.create_annotated_heatmap(z=conf_matrix, x=['boa', 'ruim'],
                                  y=['boa', 'ruim'],
                                  colorscale='Greens')

# Adicionar rótulos
fig.update_layout(title='Matriz de Confusão',
                  xaxis_title='Predito',
                  yaxis_title='Real')

# Exibir o gráfico
fig.show()
