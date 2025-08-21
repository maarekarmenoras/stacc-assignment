import pandas as pd
from sqlalchemy import create_engine

# read data
url = 'https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv'
iris = pd.read_csv(url)

# remove duplicates
iris = iris.drop_duplicates()

# Remove outliers via IQR filtering
lim = pd.DataFrame()
cols = iris.select_dtypes('number').columns

for species in ['setosa', 'versicolor', 'virginica']:
    df_sub = iris[iris['species'] == species].loc[:, cols]

    iqr = df_sub.quantile(0.75) - df_sub.quantile(0.25)
    lim = pd.concat([lim, ((df_sub - df_sub.median()) / df_sub).abs() < 2.2])
    
iris.loc[:, cols] = iris.loc[:, cols].where(lim, None)
iris = iris.dropna()

#that did not remove anything, so I guess there were no significant outliers

iris = iris.reset_index()

# write data into database
engine = create_engine("postgresql+psycopg2://postgres:gUPJELPaONJz8How@iris-postgres:5432/iris")
conn = engine.connect()
iris.to_sql(name='iris', con=conn, if_exists='fail', chunksize=10000)
print('Iris dataset successfully added to database.')
