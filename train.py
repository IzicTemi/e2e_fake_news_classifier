#!/usr/bin/python

# pylint: disable=import-error
# pylint: disable=no-name-in-module
# pylint: disable=redefined-outer-name
# pylint: disable=too-many-arguments
# pylint: disable=too-many-locals

import os
import re
import pickle
import string
import argparse
from pathlib import Path
from datetime import datetime

import nltk
import numpy as np
import mlflow
import optuna
import pandas as pd
import tensorflow as tf
from bs4 import BeautifulSoup
from prefect import flow, task, get_run_logger
from nltk.corpus import stopwords
from pandarallel import pandarallel
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from optuna.integration import TFKerasPruningCallback
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ReduceLROnPlateau, LearningRateScheduler
from tensorflow.keras.preprocessing import text, sequence
from tensorflow.keras.optimizers.schedules import ExponentialDecay


@task
def read_data(path):
    logger = get_run_logger()
    logger.info("Loading data... ")
    true = pd.read_csv(f"{path}/True.csv")
    false = pd.read_csv(f"{path}/Fake.csv")

    true['category'] = 1
    false['category'] = 0

    df = pd.concat([true, false])  # Merging the 2 datasets

    df['text'] = df['title'] + "\n" + df['text']
    del df['title']
    del df['subject']
    del df['date']
    logger.info("Successful!")
    return df


def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()


def remove_between_square_brackets(text):
    # pylint: disable=anomalous-backslash-in-string
    return re.sub('\[[^]]*\]', '', text)


def remove_stopwords(text):
    stop = set(stopwords.words('english'))
    punctuation = list(string.punctuation)
    stop.update(punctuation)
    final_text = []
    for i in text.split():
        if i.strip().lower() not in stop:
            final_text.append(i.strip())
    return " ".join(final_text)


def denoise_text(text):
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    text = remove_stopwords(text)
    return text


@task
def clean_split_data(df):
    pandarallel.initialize()
    logger = get_run_logger()
    logger.info("Cleaning and Splitting the Data")
    df['text'] = df['text'].parallel_apply(denoise_text)
    x_train, x_test, y_train, y_test = train_test_split(
        df.text, df.category, random_state=0
    )
    logger.info("Done")
    return x_train, x_test, y_train, y_test


@task
def tokenize(x_train, x_test, max_features, maxlen):
    logger = get_run_logger()
    logger.info("Tokenizing... ")
    tokenizer = text.Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(x_train)
    tokenized_train = tokenizer.texts_to_sequences(x_train)
    x_train = sequence.pad_sequences(tokenized_train, maxlen=maxlen)
    tokenized_test = tokenizer.texts_to_sequences(x_test)
    X_test = sequence.pad_sequences(tokenized_test, maxlen=maxlen)
    logger.info("Done")
    return x_train, X_test, tokenizer


def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')


@task
def get_glove_embedding(EMBEDDING_FILE, tokenizer, max_features):
    logger = get_run_logger()
    logger.info("Creating glove embedding matrix... ")
    with open(EMBEDDING_FILE, 'r', encoding='utf8') as f_out:
        embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in f_out)
    all_embs = np.stack(embeddings_index.values())
    emb_mean, emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]
    word_index = tokenizer.word_index
    nb_words = min(max_features, len(word_index))

    embedding_matrix = embedding_matrix = np.random.normal(
        emb_mean, emb_std, (nb_words, embed_size)
    )
    for word, i in word_index.items():
        if i >= max_features:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    logger.info("Done")
    return embedding_matrix


def create_lstm_model(trial, max_features, embed_size, embedding_matrix, maxlen):
    model = Sequential()
    model.add(
        Embedding(
            max_features,
            output_dim=embed_size,
            weights=[embedding_matrix],
            input_length=maxlen,
            trainable=False,
        )
    )
    dropout_rate_1 = trial.suggest_float("lstm_dropout", 0.0, 0.3)
    mlflow.log_param("dropout_lstm_layer_1", dropout_rate_1)
    model.add(
        LSTM(
            units=128,
            return_sequences=True,
            recurrent_dropout=dropout_rate_1,
            dropout=dropout_rate_1,
        )
    )
    dropout_rate_2 = trial.suggest_float("lstm_dropout_2", 0.0, 0.2)
    mlflow.log_param("dropout_lstm_layer_2", dropout_rate_2)
    model.add(LSTM(units=64, recurrent_dropout=dropout_rate_2, dropout=dropout_rate_2))
    activation_1 = trial.suggest_categorical("activation", ["relu", "selu", "elu"])
    mlflow.log_param("activation_1", activation_1)
    model.add(Dense(units=32, activation=activation_1))
    model.add(Dense(1, activation='sigmoid'))
    # lr = trial.suggest_uniform("lr", 1e-5, 1e-1)
    # mlflow.log_param("learning_rate", lr)
    mlflow.log_artifact("./save/tokenizer.bin")
    model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])
    return model


def objective(
    trial,
    x_train,
    y_train,
    batch_size,
    X_test,
    y_test,
    epochs,
    max_features,
    embed_size,
    embedding_matrix,
    maxlen,
):
    # pylint: disable=unused-variable

    # Clear clutter from previous session graphs.
    tf.keras.backend.clear_session()

    # Generate our trial model.
    model = create_lstm_model(trial, max_features, embed_size, embedding_matrix, maxlen)

    # Fit the model on the training data.
    model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        epochs=epochs,
        callbacks=[TFKerasPruningCallback(trial, "val_loss")],
    )

    # learning rate scheduler
    scheduler = ExponentialDecay(1e-3, 400 * ((len(x_train) * 0.8) / batch_size), 1e-5)
    lr = LearningRateScheduler(scheduler, verbose=0)

    # Evaluate the model accuracy on the validation set.
    score = model.evaluate(X_test, y_test, verbose=0)
    mlflow.log_metric("accuracy", score[1])
    return score[1]


@task
def train(
    x_train,
    y_train,
    batch_size,
    x_test,
    y_test,
    epochs,
    max_features,
    embed_size,
    embedding_matrix,
    maxlen,
    n,
):
    # pylint: disable=unnecessary-lambda-assignment
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.HyperbandPruner(),
    )
    func = lambda trial: objective(
        trial,
        x_train,
        y_train,
        batch_size,
        x_test,
        y_test,
        epochs,
        max_features,
        embed_size,
        embedding_matrix,
        maxlen,
    )
    study.optimize(func, n_trials=n)

    return study


def train_best_model(
    study,
    x_train,
    y_train,
    batch_size,
    x_test,
    y_test,
    epochs,
    max_features,
    embed_size,
    embedding_matrix,
    maxlen,
):
    logger = get_run_logger()
    logger.info("Starting optimzation Study")
    learning_rate_reduction = ReduceLROnPlateau(
        monitor='val_accuracy', patience=2, verbose=1, factor=0.5, min_lr=0.00001
    )
    with mlflow.start_run():
        model = Sequential()
        model.add(
            Embedding(
                max_features,
                output_dim=embed_size,
                weights=[embedding_matrix],
                input_length=maxlen,
                trainable=False,
            )
        )
        dropout_rate_1 = study.best_trial.params["lstm_dropout"]
        mlflow.log_param("dropout_lstm_layer_1", dropout_rate_1)
        model.add(
            LSTM(
                units=128,
                return_sequences=True,
                recurrent_dropout=dropout_rate_1,
                dropout=dropout_rate_1,
            )
        )
        dropout_rate_2 = dropout_rate_1 = study.best_trial.params["lstm_dropout_2"]
        mlflow.log_param("dropout_lstm_layer_2", dropout_rate_2)
        model.add(
            LSTM(units=64, recurrent_dropout=dropout_rate_2, dropout=dropout_rate_2)
        )
        activation_1 = study.best_trial.params["activation"]
        mlflow.log_param("activation_1", activation_1)
        model.add(Dense(units=32, activation=activation_1))
        model.add(Dense(1, activation='sigmoid'))
        mlflow.log_artifact("./save/tokenizer.bin")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(lr=0.01),
            loss='binary_crossentropy',
            metrics=['accuracy'],
        )
        model.fit(
            x_train,
            y_train,
            batch_size=batch_size,
            validation_data=(x_test, y_test),
            epochs=epochs,
            callbacks=[learning_rate_reduction],
        )

    return model


@task
def register_best_model(EXPT_NAME, MLFLOW_TRACKING_URI, model_name):
    # pylint: disable=too-many-statements
    logger = get_run_logger()
    logger.info("Checking run and Registering best model to Production")
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    experiment = client.get_experiment_by_name(EXPT_NAME)

    # Get best run from current experiment
    runs = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=1,
        order_by=["metrics.val_accuracy DESC"],
    )

    # Get run id of best run
    cur_run_id = runs[0].info.run_id

    # Get latest registered model versions
    try:
        client.get_latest_versions(name=model_name)
    except mlflow.exceptions.RestException:
        client.create_registered_model(
            model_name
        )  # create registered model if not exist

    # Check if model currently in production
    if (
        len(client.get_latest_versions(name=model_name, stages=["Production"])) != 0
    ):  # if yes, compare run val_accuracy to model val_accuracy
        prod_ver = client.get_latest_versions(name=model_name, stages=["Production"])[0]
        prod_run_id = prod_ver.run_id
        prod_acc = client.get_metric_history(prod_run_id, 'val_accuracy')[0].value
        run_acc = client.get_metric_history(cur_run_id, 'val_accuracy')[0].value
        if (
            run_acc > prod_acc
        ):  # if run better than production model, register new model and move to production
            logger.info('Registering new model to Production ...')
            mlflow.register_model(
                model_uri=f"runs:/{cur_run_id}/model", name=model_name
            )
            client.transition_model_version_stage(
                name=model_name,
                version=client.get_latest_versions(name=model_name, stages=["None"])[
                    0
                ].version,
                stage="Production",
                archive_existing_versions=False,
            )
            logger.info(
                'Moving previous model to Staging ...'
            )  # move previous prod model to staging
            client.transition_model_version_stage(
                name=model_name,
                version=prod_ver.version,
                stage="Staging",
                archive_existing_versions=False,
            )
        # else if production better than run, check if model in staging
        elif (
            len(client.get_latest_versions(name=model_name, stages=["Staging"])) != 0
        ):  # if yes, compare run val_accuracy to model val_accuracy
            stag_ver = client.get_latest_versions(name=model_name, stages=["Staging"])[
                0
            ]
            stag_run_id = stag_ver.run_id
            stag_acc = client.get_metric_history(stag_run_id, 'val_accuracy')[0].value
            if (
                run_acc > stag_acc
            ):  # if run better than staging model, register new model and move to staging
                logger.info('Registering new model to Staging ...')
                mlflow.register_model(
                    model_uri=f"runs:/{cur_run_id}/model", name=model_name
                )
                client.transition_model_version_stage(
                    name=model_name,
                    version=client.get_latest_versions(
                        name=model_name, stages=["None"]
                    )[0].version,
                    stage="Staging",
                    archive_existing_versions=False,
                )
                client.transition_model_version_stage(  # remove previous model from staging
                    name=model_name,
                    version=stag_ver.version,
                    stage="None",
                    archive_existing_versions=False,
                )
        else:  # if models in production and staging are both better, do nothing
            logger.info("Models in Production are better.")

    elif (
        len(client.get_latest_versions(name=model_name, stages=["Production"])) == 0
        and len(client.get_latest_versions(name=model_name, stages=["Staging"])) != 0
    ):  # run if model not in production but in staging
        stag_ver = client.get_latest_versions(name=model_name, stages=["Staging"])[0]
        stag_run_id = stag_ver.run_id
        stag_acc = client.get_metric_history(stag_run_id, 'val_accuracy')[0].value
        run_acc = client.get_metric_history(cur_run_id, 'val_accuracy')[0].value
        if (
            run_acc > stag_acc
        ):  # if run better than staging model, register new model and move to production
            logger.info('Registering model to Production ...')
            mlflow.register_model(
                model_uri=f"runs:/{cur_run_id}/model", name=model_name
            )
            client.transition_model_version_stage(
                name=model_name,
                version=client.get_latest_versions(name=model_name, stages=["None"])[
                    0
                ].version,
                stage="Production",
                archive_existing_versions=False,
            )
        else:  # promote model in staging to production if better than run
            logger.info(
                'Promoting previous model to Production and Registering new model to Staging ...'
            )
            mlflow.register_model(
                model_uri=f"runs:/{cur_run_id}/model", name=model_name
            )
            client.transition_model_version_stage(
                name=model_name,
                version=client.get_latest_versions(name=model_name, stages=["Staging"])[
                    0
                ].version,
                stage="Production",
                archive_existing_versions=False,
            )
            client.transition_model_version_stage(  # register new run to staging
                name=model_name,
                version=client.get_latest_versions(name=model_name, stages=["None"])[
                    0
                ].version,
                stage="Staging",
                archive_existing_versions=False,
            )
    else:  # if no models in production and staging
        logger.info(
            'Registering new model to Production ...'
        )  # register run to production
        mlflow.register_model(model_uri=f"runs:/{cur_run_id}/model", name=model_name)
        client.transition_model_version_stage(
            name=model_name,
            version=client.get_latest_versions(name=model_name, stages=["None"])[
                0
            ].version,
            stage="Production",
            archive_existing_versions=False,
        )


@flow(name="train model")
def study_and_train():

    logger = get_run_logger()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n_evals",
        type=int,
        default=5,
        help="the number of parameter evaluations for the optimizer to explore.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="the number of epochs to train the model",
    )

    args = parser.parse_args()
    no_evals, epochs = (args.n_evals, args.epochs)
    logger.info(f"No of optimization trials = {no_evals}")
    logger.info(f"Training for {epochs} epochs")

    nltk.download('stopwords')

    timer = datetime.now()
    timer = timer.strftime("%M-%H-%d-%m")
    EXPT_NAME = f'final-goal-{timer}'
    MLFLOW_TRACKING_URI = os.environ['MLFLOW_TRACKING_URI']
    ART_LOC = os.environ['ARTIFACT_LOC']
    model_name = os.environ['MODEL_NAME']
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.create_experiment(EXPT_NAME, artifact_location=ART_LOC)
    mlflow.set_experiment(EXPT_NAME)
    mlflow.tensorflow.autolog()

    logger.info(f"Models will be registered to {model_name}")
    batch_size = 256
    embed_size = 100
    max_features = 10000
    maxlen = 300

    path = os.getenv("DATA_PATH")
    dataframe_future = read_data.submit(path)
    split_future = clean_split_data.submit(dataframe_future)
    x_train, x_test, y_train, y_test = split_future.result()

    tokenizer_future = tokenize.submit(
        x_train, x_test, max_features, maxlen, wait_for=[split_future]
    )
    x_train, x_test, tokenizer = tokenizer_future.result()

    Path("./save").mkdir(parents=True, exist_ok=True)

    with open('./save/tokenizer.bin', 'wb') as f_out:
        pickle.dump(tokenizer, f_out)
    logger.info("Successfully saved the Tokenizer")

    EMBEDDING_FILE = f'{path}/glove.twitter.27B.100d.txt'
    embedding_matrix_future = get_glove_embedding.submit(
        EMBEDDING_FILE, tokenizer, max_features
    )
    embedding_matrix = embedding_matrix_future.result()

    train_future = train.submit(
        x_train,
        y_train,
        batch_size,
        x_test,
        y_test,
        epochs,
        max_features,
        embed_size,
        embedding_matrix,
        maxlen,
        no_evals,
        wait_for=[tokenizer_future, embedding_matrix_future],
    )

    register_best_model.submit(
        EXPT_NAME, MLFLOW_TRACKING_URI, model_name, wait_for=[train_future]
    )


if __name__ == "__main__":
    study_and_train()
