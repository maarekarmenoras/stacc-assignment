import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os
from pathlib import Path

# load environment variables
load_dotenv(dotenv_path=Path('../.env'))
POSTGRES_USER = os.getenv('POSTGRES_USER')
POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD')
POSTGRES_DB = os.getenv('POSTGRES_DB')

# read data
url = 'https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv'
iris = pd.read_csv(url)

# remove duplicates
iris = iris.drop_duplicates()

# remove outliers via IQR filtering
lim = pd.DataFrame()
cols = ['sepal_width', 'sepal_length', 'petal_width', 'petal_length']

for species in ['setosa', 'versicolor', 'virginica']:
    df_sub = iris[iris['species'] == species].loc[:, cols]
    
    # get quartiles
    q1 = df_sub.quantile(0.25)
    q3 = df_sub.quantile(0.75)
    iqr = q3 - q1
    
    # get lower and upper bounds
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    
    # find inliers
    lim = pd.concat([lim, df_sub.apply(lambda x: x.between(low[x.name], high[x.name]))])
    
iris.loc[:, cols] = iris.loc[:, cols].where(lim, None)
iris = iris.dropna()
iris = iris.reset_index()

# write data into database
engine = create_engine(f'postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}@iris-postgres:5432/{POSTGRES_DB}')
with engine.connect() as conn:
    try:
        iris.to_sql(name='iris', con=conn, if_exists='fail', chunksize=10000)
        print('Iris dataset successfully added to database.')
    except:
        print('Iris database already exists.')
