# -*- coding: utf-8 -*-
"""animes.ipynb

# Carregar os dados
"""

# Bibliotecas
import pandas as pd
import plotly.express as px

# Carregar a base
df= pd.read_csv('/content/anime.csv', encoding='utf-8')
df.head(5)

# Verificar informações
df.info()

# Verificar dados ausentes
df.isna().sum()

# Verificar dados duplicados
df.duplicated().sum()

"""# Pré-processamento"""

# Remover dados ausentes
df.dropna(inplace = True)

# Verificar dados ausentes
df.isna().sum()

class AnimePreProcessamento:
    # Construtor
    def __init__(self, df):
        self.df = df

    # Método para renomear as colunas
    def renomear_colunas_portugues(self):
        mapeamento_colunas = {
            'anime_id': 'ID',
            'name': 'nome',
            'genre': 'genero',
            'type': 'tipo',
            'episodes': 'episodios',
            'rating': 'avaliacao',
            'members': 'comunidade'
        }

        # Renomear as colunas
        self.df = self.df.rename(columns=mapeamento_colunas)

    # Método para contar os gêneros
    def contar_generos(self, string_generos):
        # Verificar se a string de gêneros não é nula
        if pd.isnull(string_generos):
            return 0

        # Dividir a string em uma lista de gêneros
        lista_generos = string_generos.split(', ')

        # Retornar o número de gêneros
        return len(lista_generos)

    # Método para criar coluna para a quantidade de gêneros dos animes
    def adicionar_coluna_numero_generos(self):
        self.df['numero_generos'] = self.df['genero'].apply(self.contar_generos)

# Instânciar o objeto
anime_processor = AnimePreProcessamento(df)

# Aplicar os métodos
anime_processor.renomear_colunas_portugues()
anime_processor.adicionar_coluna_numero_generos()

# Atualizar o df
df = anime_processor.df
df.head(5)

"""# Análises Descritivas"""

# Remover ID
df.drop('ID', inplace = True, axis = 1)

# Transformar dados desconhecidos em NaN
df['episodios'] = pd.to_numeric(df['episodios'], errors='coerce')

# Estatísticas descritivas
df.describe().round(2)

# Criar colunas numéricas
colunas_numericas = ['episodios', 'avaliacao', 'comunidade', 'numero_generos']

# Contar o total de gêneros
total_generos = df['genero'].str.split(', ', expand=True).stack().nunique()

# Criar nova coluna com o primeiro gênero
df['primeiro_genero'] = df['genero'].str.split(',').str[0]

# Distribuição (histogramas)
for col in colunas_numericas:
   fig =  px.histogram(df, x= col,
                            labels={'value': 'Valor'},
                            title=f'Histograma {col}')

   # Ajustes
   fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                     plot_bgcolor='rgba(0,0,0,0)', title_x=0.5)

   # Mostrar os histogramas
   fig.show()

# Distribuição dos Gêneros
fig = px.bar(df['genero'].str.split(', ', expand=True).stack().value_counts(),
             x=df['genero'].str.split(', ', expand=True).stack().value_counts().index,
             y=df['genero'].str.split(', ', expand=True).stack().value_counts().values,
             text_auto = True,
             labels={'x':'Gênero', 'y':'Número de Animes'},
             title=f'Distribuição de Gêneros de Animes ({total_generos} Gêneros)')

# Ajustes
fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                  plot_bgcolor='rgba(0,0,0,0)', title_x=0.5)
fig.update_traces(textposition="outside")

# Mostrar o gráfico
fig.show()

# Distribuição dos Tipos
fig = px.bar(df['tipo'].value_counts(),
             x=df['tipo'].value_counts().index,
             y=df['tipo'].value_counts().values,
             color=df['tipo'].value_counts().index,
             text_auto = True,
             labels={'x':'Tipo', 'y':'Número de Animes'},
             title='Distribuição de Tipos de Animes')

# Ajustes
fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                  plot_bgcolor='rgba(0,0,0,0)', title_x=0.5)
fig.update_traces(showlegend=False, textposition="outside")

# Mostrar o gráfico
fig.show()

"""# Análise Bivariadas"""

# Criar uma tabela dinâmica usando pivot_table com média e desvio padrão
tabela_dinamica = pd.pivot_table(df, values=colunas_numericas,
                                  index='tipo', aggfunc={'episodios': ['mean', 'std'],
                                                        'avaliacao': ['mean', 'std'],
                                                        'comunidade': ['mean', 'std'],
                                                        'numero_generos': ['mean', 'std']})

# Arredondar os valores para duas casas decimais
tabela_dinamica = tabela_dinamica.round(2)
tabela_dinamica

# Criar um dicionário para armazenar as principais entradas para cada coluna
top_entries = {col: [f'{genero} ({df.loc[df["primeiro_genero"] == genero, col].mean():.2f})'
                     for genero in df.groupby('primeiro_genero')[col].mean().nlargest(5).index]
                     for col in colunas_numericas}

# Criar uma tabela mostrando os cinco principais gêneros e suas médias para cada coluna numérica
tabela_top_generos = pd.DataFrame(top_entries)

# Exibir a tabela
tabela_top_generos

# Criar um dicionário para armazenar os top cinco animes para cada coluna
top_animes = {col: df.loc[df[col].nlargest(5).index,
              ['nome', col]].apply(lambda x: f"{x['nome']} ({x[col]:.2f})", axis=1).tolist()
              for col in colunas_numericas}

# Criar uma tabela mostrando os top cinco animes para cada coluna numérica
tabela_top_animes = pd.DataFrame(top_animes)

# Exibir a tabela
tabela_top_animes

# Variáveis numéricas
df.corr(numeric_only= True, method= 'spearman')
