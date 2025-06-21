import pandas as pd
import pickle
from pipeline import create_pipeline
from sklearn.model_selection import train_test_split

# Carrega os dados
df = pd.read_csv("adult.csv", na_values=['#NAME?'])
X = df.drop("income", axis=1)
y = df["income"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Cria e treina o pipeline
pipe = create_pipeline(X_train)
pipe.fit(X_train, y_train)

# Serializa o pipeline treinado
with open("model/pipeline.pkl", "wb") as f:
    pickle.dump(pipe, f)
