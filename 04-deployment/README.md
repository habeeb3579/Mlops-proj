# create pipenv install scikit-learn==1.5.1 flask --python=3.12.7

# pipenv shell

you can check with

```bash
pipenv --venv
```

copy the path add /bin/python and put in the select python interpreter

# PS1="> "

install gunicorn for production server

# pipenv install gunicorn

gunicorn --bind=0.0.0.0:9696 predict:app

python test.py

install requests in dev mode
pipenv install --dev requests

build docker image

```bash
docker build -t ride-duration-prediction-service:v1 .
```

pull and run the image

```bash
docker run -it --rm -p 9696:9696 ride-duration-prediction-service:v1
```

```python
python test.py
```

### use this to set config file to use for a specific repo

```bash
git remote set-url origin git@github.com-habeeb:username/repo-name.git
```
