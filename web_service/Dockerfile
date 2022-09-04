FROM public.ecr.aws/lambda/python:3.9-x86_64

RUN pip install -U pip
RUN pip install pipenv

COPY [ "Pipfile", "Pipfile.lock", "./" ]

ARG MLFLOW_TRACKING_URI
ARG MLFLOW_TRACKING_USERNAME
ARG MLFLOW_TRACKING_PASSWORD
ARG AWS_SECRET_ACCESS_KEY
ARG AWS_ACCESS_KEY_ID

ENV MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI}
ENV MLFLOW_TRACKING_USERNAME=${MLFLOW_TRACKING_USERNAME}
ENV MLFLOW_TRACKING_PASSWORD=${MLFLOW_TRACKING_PASSWORD}
ENV AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
ENV AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}

RUN pipenv install --system --deploy
RUN pip install https://files.pythonhosted.org/packages/72/21/86a60f75ab3ab110b24bee03bb08a1f50fd0e60c8b0b2050f49e47ba48af/tf_nightly-2.11.0.dev20220904-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

COPY [ "lambda_function.py", "prepare.py", "./" ]

RUN [ "python", "prepare.py"]

CMD [ "lambda_function.lambda_handler" ]
