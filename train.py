import os
os.environ['MLFLOW_TRACKING_PASSWORD'] = '***REMOVED***'
os.environ['MLFLOW_TRACKING_URI']="***REMOVED***"
os.environ['MLFLOW_TRACKING_USERNAME']="***REMOVED***"
os.environ['AWS_SECRET_ACCESS_KEY']='***REMOVED***'
os.environ['AWS_ACCESS_KEY_ID'] = '***REMOVED***'
os.environ['KAGGLE_USERNAME'] = '***REMOVED***'
os.environ['KAGGLE_KEY'] = '***REMOVED***'

import numpy as np
import pandas as pd
import nltk
from bs4 import BeautifulSoup
import re,string

from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split

import pickle
import tensorflow as tf
from nltk.corpus import stopwords
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Embedding,LSTM,Dropout
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing import text, sequence
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType

import optuna
import mlflow
from optuna.integration import TFKerasPruningCallback
from mlflow.entities import ViewType

import prefect
from prefect import flow, task
from prefect.task_runners import SequentialTaskRunner

EXPT_NAME = 'final-goal'
MLFLOW_TRACKING_URI=os.environ['MLFLOW_TRACKING_URI']
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPT_NAME)
mlflow.tensorflow.autolog()

@task
def read_***REMOVED***(path):
    true = pd.read_csv(f"{path}/True.csv")
    false = pd.read_csv(f"{path}/Fake.csv")

    true['category'] = 1
    false['category'] = 0

    df = pd.concat([true,false]) #Merging the 2 ***REMOVED***sets

    df['text'] = df['title'] +"\n"+ df['text'] 
    del df['title']
    del df['subject']
    del df['date']
    return df

def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)

def remove_between_square_brackets(text):
    return re.sub(r'http\S+', '', text)

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
def clean_split_***REMOVED***(df):
    df['text']=df['text'].apply(denoise_text)
    x_train,x_test,y_train,y_test = train_test_split(df.text,df.category,random_state = 0)

    return x_train,x_test,y_train,y_test

@task
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

@task
def get_glove_embedding(EMBEDDING_FILE, tokenizer, max_features):
    embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(EMBEDDING_FILE))
    all_embs = np.stack(embeddings_index.values())
    emb_mean,emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]
    word_index = tokenizer.word_index
    nb_words = min(max_features, len(word_index))
    #change below line if computing normal stats is too slow
    embedding_matrix = embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector
    
    return embedding_matrix


def create_lstm_model(trial, max_features, embed_size, embedding_matrix, maxlen):

    mlflow.start_run(experiment_id=2)
    model = Sequential()
    model.add(Embedding(max_features, output_dim=embed_size, weights=[embedding_matrix], input_length=maxlen, trainable=False))
    dropout_rate_1 = trial.suggest_float("lstm_dropout", 0.0, 0.3)
    mlflow.log_param("dropout_lstm_layer_1", dropout_rate_1)
    model.add(LSTM(units=128 , return_sequences = True , recurrent_dropout = dropout_rate_1 , dropout = dropout_rate_1))
    dropout_rate_2 = trial.suggest_float("lstm_dropout_2", 0.0, 0.2)
    mlflow.log_param("dropout_lstm_layer_2", dropout_rate_2)
    model.add(LSTM(units=64 , recurrent_dropout = dropout_rate_2 , dropout = dropout_rate_2))
    activation_1 = trial.suggest_categorical("activation", ["relu", "selu", "elu"])
    mlflow.log_param("activation_1", activation_1)
    model.add(Dense(units = 32 , activation = activation_1))
    activation_2 = trial.suggest_categorical("activation_2", ["sigmoid", "softmax"])
    mlflow.log_param("activation_2", activation_2)
    model.add(Dense(1, activation=activation_2))
    # lr = trial.suggest_uniform("lr", 1e-5, 1e-1)
    # mlflow.log_param("learning_rate", lr)
    mlflow.log_artifact("./save/tokenizer.bin")
    model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])
    return model

def objective(trial, x_train, y_train, batch_size, X_test, y_test, epochs, max_features, embed_size, embedding_matrix, maxlen):

    # Clear clutter from previous session graphs.
    tf.keras.backend.clear_session()


    # Generate our trial model.
    model = create_lstm_model(trial, max_features, embed_size, embedding_matrix, maxlen)

    # Fit the model on the training ***REMOVED***.
    model.fit(x_train, y_train, batch_size = batch_size , validation_***REMOVED*** = (X_test,y_test) , epochs = epochs , callbacks=[TFKerasPruningCallback(trial, "val_loss")])
            
    # learning rate scheduler
    scheduler = ExponentialDecay(1e-3, 400*((len(x_train)*0.8)/batch_size), 1e-5)
    lr = LearningRateScheduler(scheduler, verbose=0)

    # Evaluate the model accuracy on the validation set.
    score = model.evaluate(X_test, y_test, verbose=0)
    mlflow.log_metric("accuracy", score[1])
    mlflow.end_run()
    return score[1]

@task
def train(x_train, y_train, batch_size, x_test, y_test, epochs, max_features, embed_size, embedding_matrix, maxlen): 
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.HyperbandPruner())
    func = lambda trial: objective(trial, x_train, y_train, batch_size, x_test, y_test, epochs, max_features, embed_size, embedding_matrix, maxlen)
    study.optimize(func, n_trials=5)

@task
def register_best_model():
    experiment = client.get_experiment_by_name(EXPT_NAME)

    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    runs = client.search_runs(
    experiment_ids=experiment.experiment_id,
    filter_string="metrics.accuracy > 0.9",
    run_view_type=ViewType.ACTIVE_ONLY,
    max_results=2,
    order_by=["metrics.accuracy DESC"]
    )
    i = 0
    for run in runs:
        i = i+1
        mlflow.register_model( 
                model_uri = f"runs:/{run.info.run_id}/model",
                name = f'fake-news-detect-{i}'
        )
        model_name = "fake-news-detect-1"
    latest_versions = client.get_latest_versions(name=model_name)
    new_stage = "production"

    for version in latest_versions:
        model_version = version.version
        old_stage = version.current_stage

        
    if old_stage != 'production':
        client.transition_model_version_stage(
            name=model_name,
            version=model_version,
            stage=new_stage,
            archive_existing_versions=False
        )

    model_name = "fake-news-detect-2"
    latest_versions = client.get_latest_versions(name=model_name)
    new_stage = "staging"

    for version in latest_versions:
        model_version = version.version
        old_stage = version.current_stage

    if old_stage != 'staging':
        client.transition_model_version_stage(
            name=model_name,
            version=model_version,
            stage=new_stage,
            archive_existing_versions=False
        )

@flow(task_runner=SequentialTaskRunner()) 
def main():
    batch_size = 256
    epochs = 3
    embed_size = 100
    max_features = 10000
    maxlen = 300

    path = "./***REMOVED***"
    ***REMOVED***frame = read_***REMOVED***(path)
    x_train,x_test,y_train,y_test = clean_split_***REMOVED***(***REMOVED***frame)
    x_train, x_test, tokenizer = tokenize(x_train, x_test, max_features, maxlen)

    with open('./save/tokenizer.bin', 'wb') as f_out:
        pickle.dump(tokenizer, f_out)
    EMBEDDING_FILE = f'{path}/glove.twitter.27B.100d.txt'
    embedding_matrix = get_glove_embedding(EMBEDDING_FILE, tokenizer, max_features)
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience = 2, verbose=1,factor=0.5, min_lr=0.00001)
    train(x_train, y_train, batch_size, x_test, y_test, epochs, max_features, embed_size, embedding_matrix, maxlen)
    register_best_model()

from prefect.deployments import Deployment
from prefect.orion.schemas.schedules import CronSchedule
from prefect.flow_runners import SubprocessFlowRunner

Deployment(
    flow=main,
    name="model_training",
    schedule=CronSchedule(cron="0 9 15 * *"),
    flow_runner=SubprocessFlowRunner(),
)