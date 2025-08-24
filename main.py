from fastapi import FastAPI
import skops.io as sio
from sklearn.svm import SVC

def load_model():
    unknown_types = sio.get_untrusted_types(file='iris_classifier.skops')
    return sio.load('iris_classifier.skops', trusted=unknown_types)

def svc_classify(model, sepal_length: float, sepal_width: float, petal_length: float, petal_width: float):
    return model.predict([[sepal_length, sepal_width, petal_length, petal_width]])[0]

app = FastAPI()
iris_model = load_model()

@app.get('/classify/')
async def classify_iris(sepal_length: float, sepal_width: float, petal_length: float, petal_width: float):
    species = svc_classify(iris_model, sepal_length, sepal_width, petal_length, petal_width)
    return {'sepal_length': sepal_length, 'sepal_width': sepal_width, 'petal_length': petal_length, 'petal_width': petal_width, 'species': 'species'}

@app.get("/")
async def root():
    return {"message": "Hello World"}
