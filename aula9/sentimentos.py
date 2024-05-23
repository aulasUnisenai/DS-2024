
# Pré-processamento


import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import warnings
import numpy as np
warnings.filterwarnings('ignore')
import re
import spacy

# Importar módulos
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('vader_lexicon')

df = pd.read_excel('/content/letras.xlsx')

# Carregar o modelo de língua em inglês do spaCy
nlp = spacy.load("en_core_web_sm")

def preprocess_text(text):
    # Remover caracteres especiais e números, deixando apenas palavras
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Converter o texto para letras minúsculas
    text = text.lower()

    # Tokenizar o texto com spaCy
    doc = nlp(text)

    # Lematizar as palavras e remover stop words
    lemmatized_text = ' '.join([token.lemma_ for token in
                                doc if token.text not in
                                stopwords.words('english') and
                                len(token) > 3])

    return lemmatized_text

df['letrasLimpas'] = df['letras'].apply(preprocess_text)
df

# Remover os caracteres de quebra de linha
df['letrasLimpas'] = df['letrasLimpas'].str.replace('\n', ' ')

"""# Análise de Sentimentos"""

!pip install --upgrade plotly
!pip install -U kaleido

from nltk.sentiment.vader import SentimentIntensityAnalyzer
import plotly.offline as py
import plotly.graph_objs as go
import plotly
import kaleido

sid = SentimentIntensityAnalyzer()
sentimentos = df.apply(lambda r: sid.polarity_scores(r['letras']), axis=1)

# Adicionar os scores à base original.
d = pd.DataFrame(list(sentimentos))
df = df.join(d)
df.dropna(inplace=True)

# Verificar df
df.head(5)

# Selecionar a década desejada (por exemplo, 1990)
data_desejada = 2000

# Filtrar o DataFrame para a década desejada e calcular a média de 'compound' por artista
dados_ordenados = (df[df['decada'] == data_desejada]
                   .groupby('artista')['compound'].mean()
                   .sort_values(ascending=True))

# Determinar as cores com base nos valores
cores = np.where(dados_ordenados < 0, 'red', 'blue')

# Criar o gráfico de barras
trace = go.Bar(x=dados_ordenados.index,
               y=dados_ordenados.values,
               name='Sentimento',
               marker={'color': cores})
data = [trace]

# Configurar o layout do gráfico com o template escuro e sem grades
layout = go.Layout(
    title=f'Variação por artistas (sentimentos) - Década de {data_desejada}',
    xaxis={'title': 'Artistas', 'tickangle': -90, 'showgrid': False},
    yaxis={'title': 'Sentimento', 'showgrid': False}
)

# Criar a figura e mostrar o gráfico
fig = go.Figure(data=data, layout=layout)
fig.show()

# Calcular a média do sentimento por década
sentimento_decada = df.groupby('decada')['compound'].mean()

# Criar o traço do gráfico de linha
trace = go.Scatter(x=sentimento_decada.index, y=sentimento_decada.values,
                   mode='lines',
                   name='Sentimento',
                   line=dict(color='blue'))

data = [trace]

# Configurar o layout do gráfico
layout = go.Layout(
    title='Variação por década (sentimento)',
    titlefont=dict(family='Arial', size=22, color='black'),
    xaxis=dict(title='Década', tickangle=-90),
    yaxis=dict(title='Sentimento')
)

# Criar a figura do gráfico
fig = go.Figure(data=data, layout=layout)

# Mostrar o gráfico
fig.show()

"""# Termos"""

# Número de palavras por letra
df['contagemPalavras'] = df['letras'].str.split().str.len()
df['contagemPalavrasLimpas'] = df['letrasLimpas'].str.split().str.len()

# Contar as palavras únicas por letra
df['palavrasUnicas'] = df['letras'].apply(lambda x: len(set(x.split())))
df['palavrasUnicasLimpas'] = df['letrasLimpas'].apply(lambda x:
                                                      len(set(x.split())))

# Criar um dicionário vazio para armazenar os resultados por década
contagem_termos_por_decada = {}

# Selecionar as décadas desejadas
decadas = [1950, 1960, 1970, 1980, 1990, 2000, 2010]

# Para cada década, calcular os termos únicos por artista e armazenar no dicionário
for decada in decadas:
    # Filtrar o DataFrame original para a década atual
    df_decada = df[df['decada'] == decada]

    # Calcular os termos únicos por artista na década atual e armazenar diretamente no dicionário
    contagem_termos_por_decada[decada] = (df_decada.groupby('artista')
                        ['palavrasUnicasLimpas'].agg('sum').to_dict())

# Selecionar a década desejada
dados_decada = contagem_termos_por_decada[1960]

# Ordenar os dados pelo número de termos (palavras únicas) em ordem ascendente
dados_ordenados = sorted(dados_decada.items(), key=lambda x: x[1])

# Extrair os índices e os valores ordenados
artistas = [item[0] for item in dados_ordenados]
contagem_termos = [item[1] for item in dados_ordenados]

# Criar o traço do gráfico de barras
trace = go.Bar(x=artistas, y=contagem_termos,
               name='Termos',
               marker={'color': '#58508d'})
data = [trace]

# Configurar o layout do gráfico
layout = go.Layout(
    title='Variação de termos por artistas',
    titlefont={'family': 'Arial', 'size': 22, 'color': 'blue'},
    xaxis={'title': 'Artistas', 'tickangle': -90},
    yaxis={'title': 'Frequência de termos'}
)

# Criar a figura do gráfico
fig = go.Figure(data=data, layout=layout)

# Mostrar o gráfico
fig.show()

# Calcular a média de palavras únicas por década
decada = df.groupby('decada')['palavrasUnicasLimpas'].mean()

# Criar o traço do gráfico de linha
trace = go.Scatter(x=decada.index, y=decada.values,
                   mode='lines',
                   name='Termos',
                   line=dict(color='blue'))

data = [trace]

# Configurar o layout do gráfico
layout = go.Layout(
    title='Variação por década (termos)',
    titlefont=dict(family='Arial', size=22, color='black'),
    xaxis=dict(title='Década', tickangle=-90),
    yaxis=dict(title='Termos')
)

# Criar a figura do gráfico
fig = go.Figure(data=data, layout=layout)

# Mostrar o gráfico
fig.show()