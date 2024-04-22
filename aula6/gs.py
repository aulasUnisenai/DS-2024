# -*- coding: utf-8 -*-

# Manipulação de dados
import pandas as pd
# Visualização
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
# Pré-processamento
from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler
)
from sklearn.pipeline import Pipeline
# Treinamento
from sklearn.model_selection import(
    train_test_split,
    GridSearchCV)
# Avaliação
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,)
# Algoritmos
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# Carregar os dados
df = pd.read_excel('/content/renal.xlsx')
df.head(5)

# Remover ID
df.drop('id', axis =1, inplace = True)

# Alterar o tipo dos dados
cols = ['vcc', 'cgb', 'cgv']

for col in cols:
  df[col] = pd.to_numeric(df[col], errors='coerce')

df.info()

# Selecionar as colunas numéricas
colunas_numericas = df.select_dtypes(include = ['int', 'float']).columns
colunas_numericas

# Substituit NaN pela mediana
df.fillna(df.median(numeric_only= True), inplace = True)

df.isna().sum()

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

"""# Pré-processamento"""

# Criar X e y
X = df.iloc[:,: -1]
y = df.iloc[:,-1]

# Codificar alvo
le = LabelEncoder()
y = le.fit_transform(y)

# Armazenar os valores reais de y
labels = list(le.classes_)
labels

# Criar o pipeline para escalonar os dados
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('modelo', None)
])

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state = 42)

# Definir uma grade de hiperparâmetros
param_grid = [
    {
        'modelo': [RandomForestClassifier(random_state =42)],
        'modelo__n_estimators': [100, 150, 200, 300],
        'modelo__max_depth': [None, 10, 20, 30],
        'modelo__min_samples_split': [2,5,10],
        'modelo__min_samples_leaf': [1,2,4],
        'modelo__criterion': ['gini','entropy']
    },

    {
        'modelo': [KNeighborsClassifier()],
        'modelo__n_neighbors': [3, 5, 7, 9, 11],
        'modelo__weights': ['uniform', 'distance']
    },

    {
        'modelo': [XGBClassifier(random_state = 42)],
        'modelo__n_estimators': [50,100,150,200],
        'modelo__max_depth': [3,5,7,10],
        'modelo__learning_rate': [0.01, 0.1, 0.2],
        'modelo__subsample': [0.6, 0.8, 1.0],
        'modelo__colsample_bytree': [0.6, 0.8, 1.0],
        'modelo__gama': [0,0.1, 0.2],
        'modelo__min_child_weigth': [1,3,5]
    }
]

# Criar o objeto Grid Search
grid_search = GridSearchCV(pipeline, param_grid, cv = 3,
                           scoring = 'accuracy',
                           verbose =1,
                           n_jobs = -1)

# Treinar
grid_search.fit(X_train, y_train)

print(f"Melhores hiperparâmetros:"
      f"{grid_search.best_params_}")

# Melhor modelo (hiperparâmetros)
melhor_modelo = grid_search.best_estimator_
melhor_modelo

# testar
y_pred = melhor_modelo.predict(X_test)

# Matriz
modelo_matriz = confusion_matrix(y_test, y_pred)

plot_confusion_matrix(modelo_matriz, labels)

# Exibir o relatório de classificação
print('Relatório de Classificação:')
print(classification_report(y_test, y_pred))