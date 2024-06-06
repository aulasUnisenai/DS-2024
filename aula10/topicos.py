# -*- coding: utf-8 -*-
"""topicos.ipynb

# Bibliotecas
"""

import pandas as pd
from tqdm import tqdm
import os
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import warnings
warnings.filterwarnings('ignore')

"""# Carregar arquivos"""

# Categorias
categorias = ['entretenimento', 'esporte','negocio',
              'politica', 'tecnologia']

# Inicializa as listas
arquivos = []
artigos = []
artigos_categorias = []

# Caminho do diretório
caminho_base = '/content/drive/MyDrive/bbc' # Informar o seu diretório

# Coleta todos os arquivos no diretório especificado
for dirname, _, files in os.walk(caminho_base):
    for file in files:
        arquivos.append(os.path.join(dirname, file))

print(f'Encontrados {len(arquivos)} arquivos')

# Verifica as categorias e lê os arquivos
for file in tqdm(arquivos):
    cat = os.path.basename(os.path.dirname(file))
    if cat in categorias:
        artigos_categorias.append(cat)
        with open(file, 'r', encoding='utf-8', errors='ignore') as f:
            artigos.append(f.read())

assert len(artigos) == len(artigos_categorias)
print(f'Lendo {len(artigos)} artigos '
     f'de {len(set(artigos_categorias))} categorias')

"""# Básico"""

# Definindo o limite mínimo de frequência
min_freq = 5

"""##BOW (Bag-of-words)
Simples representação de texto que ignora a estrutura e a ordem das palavras no texto
"""

# Bag of Words (BoW)
vetorizador_bow = CountVectorizer(stop_words='english', min_df=min_freq)
bow = vetorizador_bow.fit_transform(artigos)

df_bow = pd.DataFrame(bow.toarray(),
                      columns=vetorizador_bow.get_feature_names_out())

df_bow.head(5)

"""## TF-IDF
Ponderação usada para avaliar a importância de uma palavra em um documento em relação a um conjunto de documentos.
"""

vetorizador_tfidf = TfidfVectorizer(stop_words='english',
                                    min_df=min_freq)

tfidf = vetorizador_tfidf.fit_transform(artigos)

df_tfidf = pd.DataFrame(tfidf.toarray(),
                        columns=vetorizador_tfidf.get_feature_names_out())

df_tfidf.head(5)

"""## N-grams
Sequência contínua de n itens em um texto, onde os itens podem ser caracteres ou palavras
"""

vetorizador_ngram = CountVectorizer(ngram_range=(2, 2),
                                    stop_words='english',
                                    min_df=min_freq)

ngram = vetorizador_ngram.fit_transform(artigos)

df_ngram = pd.DataFrame(ngram.toarray(),
                        columns=vetorizador_ngram.get_feature_names_out())
df_ngram

"""# LDA (Latent Dirichlet Allocation)"""

num_topicos = 5
lda = LatentDirichletAllocation(n_components=num_topicos, random_state=42)
lda_features = lda.fit_transform(tfidf)

# Obtendo as palavras mais importantes em cada tópico
palavras_por_topico = []
for topico, pesos in enumerate(lda.components_):
    palavras = [vetorizador_tfidf.get_feature_names_out()[i]
                for i in pesos.argsort()[:-11:-1]]
    palavras_por_topico.append(palavras)

# Criando DataFrame com as palavras mais importantes em cada tópico
df_palavras_topico = pd.DataFrame(palavras_por_topico).T
df_palavras_topico.columns = [f"Topico_{i}"
                              for i in range(1, num_topicos + 1)]
df_palavras_topico