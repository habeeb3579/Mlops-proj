```
pipenv install
```

to create a pipenv

build and run image

```bash
docker build -t my-taxi-api:v1 .
docker run -p 9696:9696 my-taxi-api:v1
```

use

```bash
pipenv lock
```

to update Pipfile.lock

you can run and predict with the FastAPI by putting in your browser

```bash
http://localhost:9696/predict?year=2023&month=5
```

or use test.py

```bash
pipenv run python test.py
```
