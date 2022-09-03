FROM public.ecr.aws/lambda/python:3.9-x86_64

RUN pip install -U pip
RUN pip install pipenv

COPY [ "Pipfile", "Pipfile.lock", "./" ]

ENV MLFLOW_TRACKING_URI="***REMOVED***"
ENV MLFLOW_TRACKING_USERNAME="***REMOVED***"
ENV MLFLOW_TRACKING_PASSWORD="***REMOVED***"
ENV AWS_SECRET_ACCESS_KEY="***REMOVED***"
ENV AWS_ACCESS_KEY_ID="***REMOVED***"

RUN pipenv install --system --deploy

COPY [ "lambda_function.py", "prepare.py", "./" ]

RUN [ "python", "prepare.py"]

CMD [ "lambda_function.lambda_handler" ]