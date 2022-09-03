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
from nltk.corpus import stopwords
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from optuna.integration import TFKerasPruningCallback
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ReduceLROnPlateau, LearningRateScheduler
from tensorflow.keras.preprocessing import text, sequence
from tensorflow.keras.optimizers.schedules import ExponentialDecay


def read_***REMOVED***(path):
    print('Loading ***REMOVED***... ')
    true = pd.read_csv(f"{path}/True.csv")
    false = pd.read_csv(f"{path}/Fake.csv")
    print('Loaded ')

    true['category'] = 1
    false['category'] = 0

    df = pd.concat([true, false])  # Merging the 2 ***REMOVED***sets

    df['text'] = df['title'] + "\n" + df['text']
    del df['title']
    del df['subject']
    del df['date']
    print('df done')
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


def clean_split_***REMOVED***(df):
    print(":split")
    df['text'] = df['text'].apply(denoise_text)
    x_train, x_test, y_train, y_test = train_test_split(
        df.text, df.category, random_state=0
    )

    return x_train, x_test, y_train, y_test


def tokenize(x_train, x_test, max_features, maxlen):
    tokenizer = text.Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(x_train)
    tokenized_train = tokenizer.texts_to_sequences(x_train)
    x_train = sequence.pad_sequences(tokenized_train, maxlen=maxlen)
    tokenized_test = tokenizer.texts_to_sequences(x_test)
    X_test = sequence.pad_sequences(tokenized_test, maxlen=maxlen)
    return x_train, X_test, tokenizer


def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')


def get_glove_embedding(EMBEDDING_FILE, tokenizer, max_features):
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

    # Fit the model on the training ***REMOVED***.
    model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        validation_***REMOVED***=(X_test, y_test),
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
            validation_***REMOVED***=(x_test, y_test),
            epochs=epochs,
            callbacks=[learning_rate_reduction],
        )

    return model


def register_best_model(EXPT_NAME, MLFLOW_TRACKING_URI, model_name):

    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    experiment = client.get_experiment_by_name(EXPT_NAME)
    runs = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=1,
        order_by=["metrics.val_accuracy DESC"],
    )
    cur_run_id = runs[0].info.run_id
    try:
        client.get_latest_versions(name=model_name)
    except mlflow.exceptions.RestException:
        client.create_registered_model(model_name)

    if len(client.get_latest_versions(name=model_name, stages=["Production"])) != 0:
        prod_ver = client.get_latest_versions(name=model_name, stages=["Production"])[0]
        prod_run_id = prod_ver.run_id
        prod_acc = client.get_metric_history(prod_run_id, 'val_accuracy')[0].value
        run_acc = client.get_metric_history(cur_run_id, 'val_accuracy')[0].value
        if run_acc > prod_acc:
            print('Registering new model to Production ...')
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
            print('Moving previous model to Staging ...')
            client.transition_model_version_stage(
                name=model_name,
                version=prod_ver.version,
                stage="Staging",
                archive_existing_versions=False,
            )
        elif len(client.get_latest_versions(name=model_name, stages=["Staging"])) != 0:
            stag_ver = client.get_latest_versions(name=model_name, stages=["Staging"])[
                0
            ]
            stag_run_id = stag_ver.run_id
            stag_acc = client.get_metric_history(stag_run_id, 'val_accuracy')[0].value
            if run_acc > stag_acc:
                print('Registering new model to Staging ...')
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
                client.transition_model_version_stage(
                    name=model_name,
                    version=stag_ver.version,
                    stage="None",
                    archive_existing_versions=False,
                )
        else:
            print("Models in Production are better.")

    elif (
        len(client.get_latest_versions(name=model_name, stages=["Production"])) == 0
        and len(client.get_latest_versions(name=model_name, stages=["Staging"])) != 0
    ):
        stag_ver = client.get_latest_versions(name=model_name, stages=["Staging"])[0]
        stag_run_id = stag_ver.run_id
        stag_acc = client.get_metric_history(stag_run_id, 'val_accuracy')[0].value
        run_acc = client.get_metric_history(cur_run_id, 'val_accuracy')[0].value
        if run_acc > stag_acc:
            print('Registering model to Production ...')
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
        else:
            print(
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
            client.transition_model_version_stage(
                name=model_name,
                version=client.get_latest_versions(name=model_name, stages=["None"])[
                    0
                ].version,
                stage="Staging",
                archive_existing_versions=False,
            )
    else:
        print('Registering new model to Production ...')
        mlflow.register_model(model_uri=f"runs:/{cur_run_id}/model", name=model_name)
        client.transition_model_version_stage(
            name=model_name,
            version=client.get_latest_versions(name=model_name, stages=["None"])[
                0
            ].version,
            stage="Production",
            archive_existing_versions=False,
        )


def load_best_model(MLFLOW_TRACKING_URI, model_name):
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    prod_version = client.get_latest_versions(name=model_name, stages=["Production"])[0]
    run_id = prod_version.run_id

    mlflow.artifacts.download_artifacts(run_id=run_id, dst_path='./artifact')
    uri_path = Path.cwd().joinpath('artifact/model').as_uri()
    model = mlflow.keras.load_model(uri_path)
    return model


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n_evals",
        type=int,
        default=5,
        help="the number of parameter evaluations for the optimizer to explore.",
    )
    parser.add_argument(
        "--epochs", type=int, default=5, help="the number of epochs to train the model"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default='fake-news-detect',
        help="the name to register models",
    )
    parser.add_argument(
        "--test",
        type=str,
        default='n',
        help="should the model be returned for testing? Enter 'y' or 'n' Default is 'n'",
    )

    args = parser.parse_args()
    no_evals, epochs, model_name, testing = (
        args.n_evals,
        args.epochs,
        args.model_name,
        args.test,
    )

    print(f"No of optimization trials {no_evals}")
    print(f"Training for {epochs} epochs")
    print(f"Models would be registered to {model_name}")

    nltk.download('stopwords')

    timer = datetime.now()
    timer = timer.strftime("%M-%H-%d-%m")
    EXPT_NAME = f'final-goal-{timer}'
    MLFLOW_TRACKING_URI = os.environ['MLFLOW_TRACKING_URI']
    ART_LOC = os.environ['ARTIFACT_LOC']
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.create_experiment(EXPT_NAME, artifact_location=ART_LOC)
    mlflow.set_experiment(EXPT_NAME)
    mlflow.tensorflow.autolog()

    batch_size = 256
    embed_size = 100
    max_features = 10000
    maxlen = 300

    path = "./***REMOVED***"
    ***REMOVED***frame = read_***REMOVED***(path)
    x_train, x_test, y_train, y_test = clean_split_***REMOVED***(***REMOVED***frame)
    print("Tokenizing... ")
    x_train, x_test, tokenizer = tokenize(x_train, x_test, max_features, maxlen)

    with open('./save/tokenizer.bin', 'wb') as f_out:
        pickle.dump(tokenizer, f_out)

    print("Creating glove embedding matrix... ")
    EMBEDDING_FILE = f'{path}/glove.twitter.27B.100d.txt'
    embedding_matrix = get_glove_embedding(EMBEDDING_FILE, tokenizer, max_features)

    print("Starting optimzation Study")
    train(
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
    )

    print("Checking run and Registering best model to Production")
    register_best_model(EXPT_NAME, MLFLOW_TRACKING_URI, model_name)

    if testing == 'y':
        print("Loading model for testing... ")
        model = load_best_model(MLFLOW_TRACKING_URI, model_name)
        print('Done')
        return model, tokenizer


if __name__ == "__main__":
    main()
