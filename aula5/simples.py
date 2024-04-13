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
# Algoritmo
from sklearn.linear_model import LinearRegression
# Pré-processamento
from sklearn.model_selection import train_test_split
# Métricas
from sklearn.metrics import mean_squared_error, r2_score

"""# Dados"""

# Carregar a base
df = pd.read_csv('/content/notas.csv')
df.head(5)

# Renomear colunas
df.rename(columns={'Hours': 'horas',
                   'Scores': 'nota'}, inplace=True)

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

# Gráfico de dispersão
fig= ex.scatter(data_frame=df,
               x="horas",
               y="nota",
               size="horas",
               trendline='ols',
               template = 'plotly_dark')
fig.show()

"""# Regressão"""

# Criar X e y
X = df.iloc[:,:-1].values
y = df.iloc[:,-1].values

# Dividir em treino e teste
X_train,X_test,y_train,y_test=train_test_split(X,y,
                                               test_size=0.2,
                                               random_state=42)

# Treinar
linear =LinearRegression()
linear.fit(X_train,y_train)

# Testar
y_pred = linear.predict(X_test)

# Avaliar o modelo
mse = mean_squared_error(y_test,y_pred)
r2  = r2_score(y_test,y_pred)
print(f"MSE (Teste): {mse.round(2)}\n"
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

# Coeficientes e intercepto da regressão
coeficientes = linear.coef_
intercepto = linear.intercept_

# Criar equação
equacao_regressao = (
    f' notas = {intercepto:.2f} + '
    f'{coeficientes[0]:.2f} * horas'
)

equacao_regressao