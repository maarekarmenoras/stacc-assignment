import pandas as pd
from sqlalchemy import create_engine
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import skops.io as sio
from dotenv import load_dotenv
import os
from pathlib import Path

# load environment variables
load_dotenv(dotenv_path=Path('../.env'))
POSTGRES_USER = os.getenv('POSTGRES_USER')
POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD')
POSTGRES_DB = os.getenv('POSTGRES_DB')

# read data from database
engine = create_engine(f'postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}@iris-postgres:5432/{POSTGRES_DB}')
conn = engine.connect()
iris = pd.read_sql('iris', con=conn)

# train test split
X = iris[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = iris['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=16)

# train model
svc = svm.SVC(kernel='rbf', C=7)
svc.fit(X_train, y_train)

# assess model accuracy
accuracy = accuracy_score(y_test, svc.predict(X_test))
print(f'Model Accuracy: {accuracy:.2f}')

# export model
sio.dump(svc, 'iris_classifier.skops')
print('Iris species classification model has been successfully exported.')
