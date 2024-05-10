# -*- coding: utf-8 -*-
"""
# Bibliotecas
"""

# Instalar biblioteca
!pip install mlxtend

# Importar bibliotecas
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

"""# Faculdade

## Dados
"""

# Gerar o conjunto de dados
dados_alunos = [
    ['Programação Orientada a Objetos',
     'Banco de Dados', 'Redes de Computadores',
     'Sistemas Operacionais'],

    ['Programação Orientada a Objetos',
     'Algoritmos e Estruturas de Dados',
     'Sistemas Operacionais'],

    ['Banco de Dados', 'Redes de Computadores', 'Sistemas Operacionais'],

    ['Programação Orientada a Objetos',
     'Banco de Dados', 'Redes de Computadores',
     'Inteligência Artificial'],

    ['Algoritmos e Estruturas de Dados',
     'Sistemas Operacionais', 'Inteligência Artificial'],

    ['Inteligência Artificial'],

    ['Programação Orientada a Objetos', 'Banco de Dados']
]

"""## Algoritmo"""

# Converter o conjunto de dados em um dataframe codificado
te = TransactionEncoder()
te_ary = te.fit_transform(dados_alunos)
df = pd.DataFrame(te_ary, columns=te.columns_)

# Verificar df
df

'''
Aplicar o algoritmo Apriori com suporte mínimo de x
e conjuntos de itens máximos de comprimento y.
'''
itens_frequentes = apriori(df,
                            min_support=0.4, # 40 %
                            max_len=5, # tamanho máximo
                            use_colnames=True)

# Gerar as regras de associação
regras = association_rules(itens_frequentes,
                          metric="confidence",
                          min_threshold=0.7)

"""## Análise"""

# Conjunto de itens frequentes
itens_frequentes

# Exibir as regras
regras.sort_values('confidence', ascending=False)

"""# Cantina - demais bases, apenas substituir os arquivos e o nome da coluna

## Dados
"""

df = pd.read_excel('/content/cantina.xlsx')
df.head(5)

# Criar o TransactionEncoder e aplicá-lo aos dados
te = TransactionEncoder()

# Criar um novo DataFrame com os dados transformados
df_encoded = te.fit_transform(df['produtos'].apply(lambda x: x.split(', ')))

df_encoded = pd.DataFrame(df_encoded,
                          columns=te.columns_)

# Verifar o novo df
df_encoded

# Aplicar o algoritmo Apriori para encontrar itens frequentes
itens_frequentes = apriori(df_encoded,
                            min_support=0.3,
                            max_len=3,
                            use_colnames=True)

# Gerar regras de associação a partir dos itens frequentes
regras = association_rules(itens_frequentes,
                           metric="confidence",
                           min_threshold=0.7)

# Conjunto de itens frequentes
itens_frequentes

# Visualizar os resultados
regras