from fastapi import FastAPI
import skops.io as sio
from sklearn.svm import SVC
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import create_engine, text
import pandas as pd
from dotenv import load_dotenv
import os
from pathlib import Path

# load environment variables
load_dotenv(dotenv_path=Path('../.env'))
POSTGRES_USER = os.getenv('POSTGRES_USER')
POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD')
POSTGRES_DB = os.getenv('POSTGRES_DB')

def load_model():
    '''
    Loads the SVC model created when the docker container was built.
    '''
    unknown_types = sio.get_untrusted_types(file='iris_classifier.skops')
    return sio.load('iris_classifier.skops', trusted=unknown_types)

def svc_classify(model: SVC, sepal_length: float, sepal_width: float, petal_length: float, petal_width: float) -> str:
    '''
    Uses the given model in order to classify an iris's species based on its attributes.

    Parameters:
    model (sklearn.svm.SVC): the SVC model loaded from load_model()
    sepal_length (float): sepal length of the iris
    sepal_width (float): sepal width of the iris
    petal_length (float): petal length of the iris
    petal_width(float): petal width of the iris

    Returns:
    str: the species the model predicted
    '''
    return model.predict([[sepal_length, sepal_width, petal_length, petal_width]])[0]

def get_similar_irises(top_n_matches: int, sepal_length: float, sepal_width: float, petal_length: float, petal_width: float) -> dict:
    '''
    Finds the n most similar irises in the dataset using cosine similarity.

    Parameters:
    top_n_matches (int): number of matches to return
    sepal_length (float): sepal length of the iris
    sepal_width (float): sepal width of the iris
    petal_length (float): petal length of the iris
    petal_width(float): petal width of the iris
    
    Returns:
    dict: dictionary containing the attribute of the n closest matches, including id in the database, sepal length, sepal width, petal length, petal width, and cosine similarity score
    '''
    with engine.connect() as conn:
        iris = pd.read_sql('iris', con=conn)

    base_iris = [sepal_length, sepal_width, petal_length, petal_width]
    iris['cosine_sim'] = iris.apply(lambda x: cosine_similarity([base_iris, x[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]])[0][1], axis=1)
    
    top_n_df = iris.sort_values(by=['cosine_sim']).head(top_n_matches)
    top_n_df = top_n_df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species', 'cosine_sim']]

    return top_n_df.to_dict()


app = FastAPI()
iris_model = load_model()
engine = create_engine(f'postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}@iris-postgres:5432/{POSTGRES_DB}')

@app.get('/classify/')
async def classify_iris(sepal_length: float, sepal_width: float, petal_length: float, petal_width: float):
    '''
    Predicts the species of an iris and retrieves top 5 matches from the original dataset.
    
    Query parameters:
    **sepal_length (float)**: sepal length of the iris
    **sepal_width (float)**: sepal width of the iris
    **petal_length (float)**: petal length of the iris
    **petal_width(float)**: petal width of the iris

    '''
    species = svc_classify(iris_model, sepal_length, sepal_width, petal_length, petal_width)
    
    top_5_similar = get_similar_irises(5, sepal_length, sepal_width, petal_length, petal_width)

    return {
            'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width,
            'species': species,
            'closest_matches': top_5_similar
            }

@app.get('/health/')
async def health():
    '''
    Healthcheck. Returns 'status': 'healthy' if the API is up and running.
    '''
    return {'status': 'healthy'}
