import pandas as pd
from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection import train_test_split

from pipeline import create_pipeline

df = pd.read_csv('adult.csv', na_values=['#NAME?']) # Ajuste o caminho conforme necessário
X = df.drop('income', axis=1)
y = df['income']

def test_pipeline_fit():
    pipe = pipeline.create_pipeline(X)  # Só passa X agora
    pipe.fit(X, y)
    check_is_fitted(pipe)

def test_pipeline_predict_shape():
    pipe = pipeline.create_pipeline(X)
    pipe.fit(X.iloc[:20], y.iloc[:20])
    y_pred = pipe.predict(X.iloc[:20])
    assert len(y_pred) == 20