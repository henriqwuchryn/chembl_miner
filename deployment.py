import pandas as pd
import sklearn. preprocessing as preproc
import numpy as np
import seaborn as sns
sns.set_theme(style='ticks')
sns.set_theme(style='whitegrid',font='liberation serif')
import matplotlib.pyplot as plt
import miscelanneous_methods as mm
import machine_learning_methods as mlm
import molecules_manipulation_methods as mmm
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
    molecules_df = pd.read_csv(f'{datasets_path}/{dataset_filename}')
except:
    if not os.path.exists(f'{datasets_path}/{dataset_filename}'):
        print('\nFile does not exist')
    else:
        print('\nInvalid file - cannot convert to dataframe')
    quit()
if not os.path.exists(results_path):
    os.makedirs(results_path)

try:
    with open (model_filename, 'rb') as file:
        model = pickle.load(file)
except:
    print('cant read model')

molecules_df = molecules_df.dropna(subset='canonical_smiles').reset_index()

mmm.calculate_fingerprint(molecules_df, 'fingerprint/PubchemFingerprinter.xml')
fingerprint = pd.read_csv('descriptors.csv')
fingerprint = fingerprint.drop('Name', axis=1)
fingerprint = mlm.scale_features(fingerprint, preproc.MinMaxScaler())

os.remove('descriptors.csv')
os.remove('descriptors.csv.log')
os.remove('molecules.smi')

model_features = model.feature_names_in_
feature_mask = np.isin(fingerprint.columns, model_features)
fingerprint = fingerprint[fingerprint.columns[feature_mask]]

y_pred = model.predict(fingerprint)
y_pred = pd.Series(y_pred, name='pIC50 predicted')

molecules_df_result = pd.concat([molecules_df, y_pred], axis=1, ignore_index=True)
print(molecules_df_result)
