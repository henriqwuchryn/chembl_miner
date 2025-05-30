import pandas as pd
import numpy as np
import seaborn as sns
sns.set_theme(style='ticks')
sns.set_theme(style='whitegrid',font='liberation serif')
import matplotlib.pyplot as plt
import miscelanneous_methods as mm
import joblib
from sklearn.ensemble import *
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import sklearn.metrics as metrics
import os
import sys
import pickle

try:
    dataset_filename = sys.argv[1]
except:
    print(
        '''\nyou must insert the dataset filename as first argument, like this:
    >python deployment.py DATASET_FILENAME.csv model_filename.pkl'''
    )
    quit()

try:
    model_filename = sys.argv[2]
except:
    print(
        '''\nyou must insert the model filename as second argument, like this:
    >python deployment.py dataset_filename.csv MODEL_FILENAME.pkl'''
    )
    quit()

results_path = f'analysis/{dataset_filename[:-4]}'
datasets_path = 'datasets'

try:
    molecules_df = pd.read_csv(f'{datasets_path}/{dataset_filename}', index_col='index')
except:
    if not os.path.exists(f'{datasets_path}/{dataset_filename}'):
        print('\nFile does not exist')
    else:
        print('\nInvalid file - cannot convert to dataframe')
    quit()
if not os.path.exists(results_path):
    os.makedirs(results_path)
