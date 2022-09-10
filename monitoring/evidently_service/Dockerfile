FROM python:3.8-slim-buster

RUN pip3 install -U pip

RUN pip3 install pipenv

WORKDIR /app

COPY [ "Pipfile", "Pipfile.lock", "./" ]

RUN pipenv install --system --deploy

COPY [ "app.py", "./" ]

CMD [ "gunicorn", "--bind=0.0.0.0:8085", "--timeout=600", "--worker-class=gevent", "app:app" ]
