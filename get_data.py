#!/usr/bin/python

import os
import zipfile
from pathlib import Path

import kaggle
from prefect import flow, task, get_run_logger

KAGGLE_USERNAME = os.getenv("KAGGLE_USERNAME")
KAGGLE_KEY = os.getenv("KAGGLE_KEY")

DATASET_KAGGLE_URL = 'clmentbisaillon/fake-and-real-news-dataset'
GLOVE_EMBEDDINGs_URL = 'icw123/glove-twitter'

DATA_PATH = os.getenv("DATA_PATH")

Path(DATA_PATH).mkdir(parents=True, exist_ok=True)


@task
def download_dataset(DATASET_KAGGLE_URL):
    '''
    Get dataset from Kaggle API
    '''
    logger = get_run_logger()
    logger.info("Downloading Dataset")
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(DATASET_KAGGLE_URL, path=f'{DATA_PATH}')


@task
def download_glove(GLOVE_EMBEDDINGs_URL):
    '''
    Get glove embeddings from Kaggle API
    '''
    logger = get_run_logger()
    logger.info("Downloading Glove embeddings")
    kaggle.api.authenticate()
    kaggle.api.dataset_download_file(
        GLOVE_EMBEDDINGs_URL,
        file_name='glove.twitter.27B.100d.txt',
        path=f'{DATA_PATH}',
    )


@task
def unzip():
    logger = get_run_logger()
    logger.info("Unzipping...")
    dir_list = os.listdir(DATA_PATH)
    for file in dir_list:
        if file.endswith('.zip'):
            with zipfile.ZipFile(f'{DATA_PATH}/{file}', 'r') as zip_ref:
                for file_unzipped in zip_ref.infolist():
                    if file_unzipped.filename not in dir_list:
                        logger.info(f"Unzipping... {file}")
                        zip_ref.extract(file_unzipped, DATA_PATH)


@task
def del_zip():
    logger = get_run_logger()
    logger.info('Deleting zip files')
    dir_list = os.listdir(DATA_PATH)
    for file in dir_list:
        if file.endswith(".zip"):
            logger.info(f'Deleting {file}')
            os.remove(Path(DATA_PATH).joinpath(file))


@flow(name="get_data")
def get_data():
    dataset = download_dataset.submit(DATASET_KAGGLE_URL)
    glove = download_glove.submit(GLOVE_EMBEDDINGs_URL)
    unzipper = unzip.submit(wait_for=[dataset, glove])
    del_zip.submit(wait_for=[unzipper])


if __name__ == '__main__':
    get_data()
