# -*- coding: utf-8 -*-
"""# Bibliotecas"""

# Manipulação de dados
import pandas as pd
# Visualização
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
# Pré-processamento
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler,
    OneHotEncoder
)
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
     f1_score
)
# Algoritmos
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

"""# Dados"""

# Carregar os dados
df = pd.read_excel('/content/spotify.xlsx')
df.head(5)

# Verificar informações
df.info()

# Verificar NaN
df.isna().sum()

# Remover NaN
df.dropna(axis = 0, inplace = True)
df.isna().sum()

# Alterar tipos
cols = ['tom', 'modo', 'tempo_assinatura']

df[cols] = df[cols].astype('object')

df.info()

# Deletar colunas
cols_deletar = ["artista", "album", "faixa", "faixa_id"]

df.drop(cols_deletar, axis=1, inplace = True)

# Colunas numéricas
colunas_numericas = df.select_dtypes(include=['int', 'float']).columns
colunas_numericas

# Colunas categóricas
colunas_categoricas = df.select_dtypes(include=['object']).columns
colunas_categoricas

"""# Funções auxiliares"""

# Função para treinar
def treinar_avaliar(X_train, y_train, X_test, y_test):
    classificadores = {
        'Random Forest': RandomForestClassifier(random_state=42),
        'Knn': KNeighborsClassifier(),
        'SVM': SVC(random_state=42),
        'GNB': GaussianNB()
    }

    resultados = []

    for nome, classificador in classificadores.items():
        classificador.fit(X_train, y_train)
        y_pred = classificador.predict(X_test)
        acuracia = accuracy_score(y_test, y_pred)
        f1Score = f1_score(y_test, y_pred, average='weighted')
        resultados.append({'Modelo': nome,
                           'Acurácia': (acuracia*100).round(2),
                           'F1 Score': (f1Score*100).round(2)})

    return pd.DataFrame(resultados)

# Função para a matriz de confusão
def plot_confusion_matrix(conf_matrix, labels):
    fig = ff.create_annotated_heatmap(z=conf_matrix[::-1],
                                      x=labels,
                                      y=labels[::-1],
                                      colorscale='blues')

    fig.update_layout(title='Matriz de Confusão',
                      xaxis=dict(title='Classe Prevista', side='bottom'),
                      yaxis=dict(title='Classe Real'))

    fig.show()

"""# EDA"""

# Descritivas
df.describe().T.round(2)

# Médias por gêneros
pd.pivot_table(df, index='genero',
               values=colunas_numericas,
               aggfunc={'mean', 'std'}).round(2)

# Correlações
corr = df.corr(numeric_only=True)
corr.style.background_gradient()

# Valores Únicos categóricos
for col in colunas_categoricas:
    print(col, df[col].unique())

# Classe para gerar os gráficos
class Graficos:
    def __init__(self, df):
        self.df = df

    def histograma(self, col1):
        fig = px.histogram(self.df,
                           x=col1, marginal="box",
                           template='plotly_dark')
        fig.show()

    def boxplot_por_genero(self, col1, col_genero='genero'):
        fig = px.box(self.df,
                     x=col_genero, y=col1,
                     color = col_genero,
                     template='plotly_dark')

        fig.update_layout(xaxis_title='Gênero', yaxis_title=col1)

        fig.show()

    def barras(self, col1):
        # Contar ocorrências de valores na coluna
        counts = self.df[col1].value_counts()

        # Criar gráfico de barras
        fig = px.bar(x=counts.index,
                     y=counts.values,
                     text_auto = True,
                     template='plotly_dark')

        fig.update_traces(textposition='outside')

        fig.update_layout(xaxis_title=col1,
                          yaxis_title='Contagem',
                          )

        fig.show()

# Criar o objeto
graficos = Graficos(df)

# Histograma
for col in colunas_numericas:
    graficos.histograma(col)

# Barras
for col in colunas_categoricas:
    graficos.barras(col)

# Boxplot gêneros
for col in colunas_numericas:
    graficos.boxplot_por_genero(col)

"""# Pré-processamento"""

# Separar X e Y
X = df.drop('genero', axis = 1)
y = df['genero']

# Codificar alvo
le = LabelEncoder()
y = le.fit_transform(y)

# Armazenar os valores da classe
labels = list(le.classes_)

# Definir colunas numéricas e categóricas
numeric_features = colunas_numericas
categorical_features = colunas_categoricas.drop('genero')

# Pré-processamento das colunas categóricas
categorical_transformer = OneHotEncoder(
                      handle_unknown='ignore')

# Pipeline para pré-processamento das features numéricas
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# Combinar os pré-processadores em um ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ],  remainder='drop')

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y, test_size=0.2,
                                                    random_state=42,
                                                    stratify=y
                                                )

# Aplicar o pré-processamento nos conjuntos de treino e teste
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Obter as colunas após o pré-processamento
transformer_names = preprocessor.named_transformers_

cat_columns = transformer_names['cat'].get_feature_names_out(
    input_features=categorical_features)

num_columns = numeric_features

# Combinar as colunas numéricas e categóricas
processed_columns = list(num_columns) + list(cat_columns)

# Exibir as colunas
print("Colunas após o pré-processamento:")
print(processed_columns)

"""# Classificação"""

# Treinar
resultados_df = treinar_avaliar(X_train_processed,
                             y_train,
                             X_test_processed,
                             y_test)

resultados_df

# Criar gráfico de comparação com subplots e adicionar as barras
fig = make_subplots(rows=2, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.1,
                    subplot_titles=("Acurácia", "F1-Score"))

# Adicionar as barras de acurácia e F1-Score
fig.add_trace(go.Bar(x=resultados_df['Modelo'],
                     y=resultados_df['Acurácia'],
                     name='Acurácia',
                     text=round(resultados_df['Acurácia'], 2),
                     textposition='auto',
                     insidetextanchor='start'),
                     row=1, col=1)

fig.add_trace(go.Bar(x=resultados_df['Modelo'],
                     y=resultados_df['F1 Score'],
                     name='F1 Score',
                     text=round(resultados_df['F1 Score'], 2),
                     textposition='auto',
                     insidetextanchor='start'),
                     row=2, col=1)

# Atualizar layout
fig.update_layout(template='plotly_dark',
                  title='Comparação de Acurácia e F1-Score',
                  yaxis=dict(title='Acurácia', tickformat=".2f"),
                  yaxis2=dict(title='F1 Score', tickformat=".2f")
                  )

# Exibir o gráfico
fig.show()

"""# Avaliação (Melhor Modelo)"""

rf = RandomForestClassifier(random_state=42)
rf.fit(X_train_processed, y_train)

# Acurácia
rf_acc = accuracy_score(y_test, rf.predict(X_test_processed))

print(f"Acurácia (treino):"
      f"{accuracy_score(y_train, rf.predict(X_train_processed))}")

print(f"Acurácia (teste): {rf_acc:.2f}")

# Matriz de confusão
rf_matriz = confusion_matrix(y_test, rf.predict(X_test_processed))

# Mostrar matriz
plot_confusion_matrix(rf_matriz, labels)

# Relatório de classificação
print(classification_report(y_test, rf.predict(X_test_processed),
                            target_names=labels))