import os

import kaggle

import zipfile

from pathlib import Path

KAGGLE_USERNAME = os.getenv("KAGGLE_USERNAME")
KAGGLE_KEY = os.getenv("KAGGLE_KEY")

DATASET_KAGGLE_URL = 'clmentbisaillon/fake-and-real-news-***REMOVED***set'
GLOVE_EMBEDDINGs_URL = 'icw123/glove-twitter'

DATA_PATH = '***REMOVED***'

def download_***REMOVED***set():
    '''
    Get ***REMOVED***set from Kaggle API
    '''
    print('Downloading Dataset')
    kaggle.api.authenticate()
    kaggle.api.***REMOVED***set_download_files(
        DATASET_KAGGLE_URL,
        path=f'{DATA_PATH}'
    )

def download_glove():
    '''
    Get glove embeddings from Kaggle API
    '''
    print('Downloading glove')
    kaggle.api.authenticate()
    kaggle.api.***REMOVED***set_download_file(
        GLOVE_EMBEDDINGs_URL,
        file_name='glove.twitter.27B.100d.txt',
        path=f'{DATA_PATH}'
        )
    

def unzip():
    print('Unzipping...')
    dir_list = os.listdir(DATA_PATH)
    for file in dir_list:
        with zipfile.ZipFile(f'{DATA_PATH}/{file}', 'r') as zip_ref:
            print(f'Unzipping {file}')
            zip_ref.extractall(DATA_PATH)

def del_zip():
    print('Deleting zip files')
    dir_list = os.listdir(DATA_PATH)
    for file in dir_list:
        if file.endswith(".zip"):
            print(f'Deleting {file}')
            os.remove(Path(DATA_PATH).joinpath(file))
    
if __name__ == '__main__':
    download_***REMOVED***set()
    download_glove()
    unzip()
    del_zip()