# -*- coding: utf-8 -*-
"""renal.ipynb

# Bibliotecas
"""

import pandas as pd
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

"""# Dados"""

df = pd.read_excel('/content/renal.xlsx',
                   sheet_name='dados')
df.head(5)

# Informações
df.info()

# Verificar NaN
df.isna().sum().sort_values(ascending = False)

# Remover id
df.drop('id', axis = 1, inplace = True)

# Alterar o tipo do dado
cols = ['vcc','cgb','cgv']

for col in cols:

   df[col] = pd.to_numeric(df[col],
                                errors='coerce')

# Selecionar colunas numéricas
df_numericas = df.select_dtypes(include=['int', 'float']).columns
df_numericas

"""# EDA"""

# Estatísticas descritivas
df.describe().T.round(2)

# Substituir NaN por mediana
df.fillna(df.median(numeric_only = True), inplace = True)

df.isna().sum()

# Função para os gráficos
def plot(col1, col2=None):
    if col2 is None:
        fig = px.histogram(df, x=col1, marginal="box",
                           template='plotly_dark')
    else:
        fig = px.scatter(df, x=col1, y=col2,
                         color="resultado",
                         template='plotly_dark')
    fig.show()

# Histograma
for col in df_numericas:
    plot(col)

# Dispersão
for i in range(len(df_numericas)):
    for j in range(i+1, len(df_numericas)):
        plot(df_numericas[i], df_numericas[j])

"""# Classificação"""

# Alterar a variável alvo
le = LabelEncoder()
df['resultado'] = le.fit_transform(df['resultado'])

# Criar X e y
X = df.drop('resultado', axis = 1)
y = df['resultado']

# Separar em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = 0.30,
                                                    random_state =42)

# Função para criar a matriz
def plot_confusion_matrix(conf_matrix):
    fig = px.imshow(conf_matrix,
                    labels=dict(x="Classe Prevista",
                                y="Classe Real"),
                    x=['Não (Previsto)', 'Sim (Previsto)'],
                    y=['Não (Real)', 'Sim (Real)'],
                    color_continuous_scale='blues',
                    text_auto=True
                    )
    fig.update_layout(title='Matriz de Confusão')
    fig.show()

"""## C4.5"""

# Treinar
c45 = DecisionTreeClassifier(criterion='entropy')
c45.fit(X_train, y_train)

# Avaliar
c45_acc = accuracy_score(y_test, c45.predict(X_test))

print(f"Acurácia (treino): {accuracy_score(y_train, c45.predict(X_train))}")
print(f"Acurácia (teste): {c45_acc:.2f}")

# Matriz de confusão
c45_matriz = confusion_matrix(y_test, c45.predict(X_test))

# Mostrar matriz
plot_confusion_matrix(c45_matriz)

# Relatório de classificação
print(classification_report(y_test, c45.predict(X_test)))

"""## CART"""

# Treinar
cart = DecisionTreeClassifier(criterion='gini')
cart.fit(X_train, y_train)

# Avaliar
cart_acc = accuracy_score(y_test, cart.predict(X_test))

print(f"Acurácia (treino): {accuracy_score(y_train, cart.predict(X_train))}")
print(f"Acurácia (teste): {cart_acc:.2f}")

# Matriz de confusão
cart_matriz = confusion_matrix(y_test, cart.predict(X_test))

# Mostrar matriz
plot_confusion_matrix(cart_matriz)

# Relatório de classificação
print(classification_report(y_test, cart.predict(X_test)))

"""## Random Forest"""

# Treinar
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

# Avaliar
rf_acc = accuracy_score(y_test, rf.predict(X_test))

print(f"Acurácia (treino): {accuracy_score(y_train, rf.predict(X_train))}")
print(f"Acurácia (teste): {rf_acc:.2f}")

# Matriz de confusão
rf_matriz = confusion_matrix(y_test, rf.predict(X_test))

# Mostrar matriz
plot_confusion_matrix(rf_matriz)

# Relatório de classificação
print(classification_report(y_test, rf.predict(X_test)))

"""# Comparar modelos"""

# Criar df para armazenar os resultados
models = pd.DataFrame({
    'Modelo' : [ 'c4.5', 'CART', 'RF'],
    'Acurácia' : [c45_acc, cart_acc, rf_acc]
})

models = models.sort_values(by = 'Acurácia', ascending = True)

# Criar gráfico de comparação
px.bar(data_frame = models, x = 'Acurácia',
        y = 'Modelo',
        template='plotly_dark',
        text_auto=True,
        title = 'Comparação')
