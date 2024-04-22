# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.model_selection import (
    cross_val_score,
    KFold,
    cross_validate)
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler
)
from sklearn.pipeline import Pipeline

df = pd.read_excel('/content/renal.xlsx')
df.head(5)

df.isna().sum()

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

"""# Treinar"""

# Criar um objeto Kfold para a validação cruzada
kf = KFold(n_splits=3,
           shuffle= True,
           random_state = 42)

# Iniciar o modelo
modelo = RandomForestClassifier(random_state = 42)

# Criar o pipeline para escalonar os dados
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('modelo', modelo)
])

# Realizar a validação cruzada
scores = cross_val_score(pipeline, X, y,
                         cv = kf,
                         scoring = 'accuracy'
                         )

# Verificar as acurácias
scores

# Verificar as médias das acurácias
media_acuracia = scores.mean() * 100
desvio_acuracia =(scores.std()).round(3)
print(f'Média da Acurácia: {media_acuracia:.2f} (+/- {desvio_acuracia})')