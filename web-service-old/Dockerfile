FROM python:3.9.7-slim

RUN pip install -U pip
RUN pip install pipenv 

WORKDIR /app

COPY [ "./Pipfile", "./Pipfile.lock", "./" ]

ENV MLFLOW_TRACKING_URI="***REMOVED***"
ENV MLFLOW_TRACKING_USERNAME="***REMOVED***"
ENV MLFLOW_TRACKING_PASSWORD="***REMOVED***"
# ENV AWS_SECRET_ACCESS_KEY="***REMOVED***"
# ENV AWS_ACCESS_KEY_ID="***REMOVED***"

RUN pipenv install --system --deploy

COPY [ "classify.py", "prepare.py", "./" ]

RUN [ "python", "prepare.py"]

EXPOSE 9696

ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:9696", "classify:app" ]