# Simple Iris Classifier API

This is a web API built using FastAPI utilising the iris dataset. When given an iris's sepal length and width as well as petal length and width, it will return a prediction for what this iris's species is, using an SVC from sklearn. It also provides 5 closest matches to the given iris from the original iris dataset. It does not currently have an UI.

## Installation and usage

1. Clone the repo.
2. Build the project using
```
docker compose up -d
```
3. Wait until the container iris-api turns healthy.
4. Go to localhost:5454
5. Try out the API using the example or fill in your own numbers.
```
http://localhost:5454/classify/?sepal_length=5.1&sepal_width=3.5&petal_length=1.4&petal_width=0.2
```
6. Look at the results in JSON.

## Documentation

Bare-bones API documentation can be found at localhost:5454/docs

## To do
Future additions to this project would include:
* A front-end built either with HTML templates of js
* The ability to add new rows to the database
