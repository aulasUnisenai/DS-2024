"""
# Bibliotecas Gerais
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

"""# Análise"""

import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose

# Carregar a base
df = pd.read_csv('/content/eletricidade.csv',
                 parse_dates=['data'], index_col='data')

df.head(5)

# Informações
df.info()

# Verificar a série
fig_original = px.line(df, x = df.index, y = 'producao')
fig_original.update_layout(title='Série Original')

# Decompor
decomposicao = sm.tsa.seasonal_decompose(df['producao'],
                                          model='additive')

# Acessar componentes
tendencia = decomposicao.trend
sazonalidade = decomposicao.seasonal
ruido = decomposicao.resid

# Tendência (direção de longo prazo dos dados)
fig_tendencia = px.line(x = tendencia.index, y = tendencia)
fig_tendencia.update_layout(title='Tendência')

# Sazonalidade (padrões repetitivos em intervalos regulares)
fig_sazonalidade = px.line(x = sazonalidade.index, y = sazonalidade)
fig_sazonalidade.update_layout(title='Sazonalidade')

# Ruído (variações aleatórias não explicadas)
fig_ruido = px.line(x = ruido.index, y = ruido)
fig_ruido.update_layout(title='Ruído')

"""# Previsão"""

from prophet import Prophet
from prophet.diagnostics import performance_metrics
from sklearn.metrics import(
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score)

df = pd.read_csv('/content/eletricidade.csv', parse_dates=['data'])
df.head(5)

# Informações
df.info()

# Ajustar o df para a estrutura da biblioteca
df = df[['data', 'producao']]
df.rename(columns={'data': 'ds', 'producao': 'y'},
          inplace=True)

# Instanciar o modelo
modelo = Prophet(interval_width=0.95)
# Realizar o treinamento
modelo.fit(df)

# Configurar o modelo para previsão
futuro  = modelo.make_future_dataframe(
    periods= 12,
    freq = 'm',
    include_history = True
)

# Realizar a previsãp
previsao = modelo.predict(futuro)

# Verificar gráfico
grafico_previsao = modelo.plot(previsao)
grafico_previsao.show()

# Unir os df (real e previstos)
df_mesclado = pd.merge(df,
                       previsao[['ds','yhat_lower','yhat','yhat_upper']],
                    on='ds').round(2)

df_mesclado.head(5)

# Avaliar o modelo
y_verdadeiro = df_mesclado['y'].values
y_previsto = df_mesclado['yhat'].values
mae = mean_absolute_error(y_verdadeiro, y_previsto)
mape = mean_absolute_percentage_error(y_verdadeiro, y_previsto)
r2 = r2_score(y_verdadeiro, y_previsto)

print(f"MAE: {mae:.2f}")
print(f"MAPE:{mape * 100:.2f}%")
print(f"R2: {r2* 100:.2f}%")

# Resumir a previsão
previsao[['ds', 'yhat_lower', 'yhat', 'yhat_upper']].tail(11)

# Criar a figura e adicionar os dados históricos e previsões
fig = go.Figure([
    go.Scatter(x=df_mesclado['ds'], y=df_mesclado['y'],
               mode='lines', name='Dados Históricos'),

    go.Scatter(x=previsao['ds'], y=previsao['yhat'],
               mode='lines', name='Previsão'),
])

# Customizar o layout do gráfico
fig.update_layout(
    template ='plotly_dark',
    title='Previsões',
    xaxis=dict(showgrid=False, title='Data'),
    yaxis=dict(showgrid=False, title='Valor')
)

# Mostrar o gráfico
fig.show()