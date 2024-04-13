# -*- coding: utf-8 -*-
"""
# Bibliotecas
"""

# Manipulação de dados
import pandas as pd
import numpy as np
# Visualização de dados
import plotly.express as ex
import plotly.graph_objects as go
# Algoritmos
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
# Pré-processamento
from sklearn.model_selection import train_test_split
# Métricas
from sklearn.metrics import mean_squared_error, r2_score

"""# Dados"""

# Carregar a base
df = pd.read_csv('/content/concreto.csv')
df.head(5)

# Verificar informações
df.info()

# Verificar NaN
df.isna().sum()

"""# EDA"""

# Estatísticas descritivas
df.describe().T.round(2)

# Correlação
corr = df.corr(numeric_only=True)
corr.style.background_gradient()

# Função para treinar
def treinar_avaliar(X_train, y_train, X_test, y_test):
    regressores = {
        'Regressão Linear': LinearRegression(),
        'Random Forest': RandomForestRegressor(random_state=42)
    }

    resultados = []

    for nome, regressor in regressores.items():
        regressor.fit(X_train, y_train)
        y_pred = regressor.predict(X_test)
        mse = mean_squared_error(y_test,y_pred)
        r2 = r2_score(y_test, y_pred)
        resultados.append({'Modelo': nome,
                           'Erro Quadrático Médio': mse.round(2),
                           'R²': (r2*100).round(2)})

    return pd.DataFrame(resultados)

"""# Regressão"""

# Criar X e y
X = df.iloc[:,:-1].values
y = df.iloc[:,-1].values

# Dividir em treino e teste
X_train,X_test,y_train,y_test=train_test_split(X,y,
                                               test_size=0.2,
                                               random_state=42)

# Treinar
resultados = treinar_avaliar(X_train,
                             y_train,
                             X_test,
                             y_test)

resultados.sort_values(by='Erro Quadrático Médio')

"""# Melhor modelo"""

rf = RandomForestRegressor(random_state = 42)
rf.fit(X_train, y_train)

# Testar
y_pred = rf.predict(X_test)

# Avaliar o modelo
mse = mean_squared_error(y_test,y_pred)
r2  = r2_score(y_test,y_pred)
print(f"Erro Quadrático Médio (Teste): {mse.round(2)}\n"
      f"R2 (Teste): {r2.round(2)}")

# Criar figura
fig = go.Figure()

# Adicionar linha de tendência (valores reais)
fig.add_trace(go.Scatter(x=y_test,
                         y=y_test,
                         mode='lines',
                         name='Real',
                         line=dict(color='blue')))

# Adicionar pontos para valores previstos
fig.add_trace(go.Scatter(x=y_test,
                         y=y_pred,
                         mode='markers',
                         name='Previsto',
                         marker=dict(color='red')))

# Layout do gráfico
fig.update_layout(title='Comparação entre Valores Reais e Previstos',
                  xaxis_title='Valores Reais',
                  yaxis_title='Valores Previstos',
                  showlegend=True)

# Exibir gráfico
fig.show()

"""# Equação"""

# Obter a importância das características
importancias = rf.feature_importances_

# Montar a equação com as características mais importantes
equacao_rf = 'Resistência = '
for i in range(len(df.columns[:-1])):
    equacao_rf += f'{importancias[i]:.2f} * {df.columns[i]} + '

# Remover o último sinal de adição
equacao_rf = equacao_rf

equacao_rf[:-3]

"""# Gráfico de Radar"""

# Nome das características
nomes_caracteristicas = df.columns[:-1]

# Criar o gráfico de radar
fig = go.Figure()

fig.add_trace(go.Scatterpolar(
      r=importancias,
      theta=nomes_caracteristicas,
      fill='toself',
      name='Importância das Características'
))

fig.update_layout(
  polar=dict(
    radialaxis=dict(
      visible=True,
      range=[0, max(importancias)]
    )),
  showlegend=True
)

fig.show()