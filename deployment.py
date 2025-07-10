import glob
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
import re

# definition of paths
try:
    deploy_filename = sys.argv[1]
    if deploy_filename[-4:] == '.csv':
        deploy_filename = deploy_filename[:-4]
except:
    print(
            '''\nyou must insert the deployment dataset filename as first argument, like this:
            >python deployment.py TEST_DATASET_FILENAME.csv model_dataset_filename'''
            )
    quit()

try:
    model_filename = sys.argv[2]
    if model_filename[-4:] == '.csv':
        model_filename = model_filename[:-4]
    if model_filename[-4:] == '.pkl':
        model_filename = model_filename[:-4]
except:
    print(
            '''\nyou must insert the model filename as second argument, like this:
            >python deployment.py deploy_dataset_filename.csv model_dataset_filename'''
            )
    quit()

results_path = f'analysis/{deploy_filename}'
datasets_path = 'datasets'

# model selection
model_files = glob.glob(f'analysis/{model_filename}/*.pkl')
model_files.sort()
model_files_df = pd.DataFrame(model_files)
print(model_files_df)
model_index:str = input('\nSelect which model you want to use by inserting the index\nNegative index will use all models\n')
try:
    model_index = int(model_index)
    if model_index >= 0:
        model_path = model_files[model_index]
except:
    print('\nPlease type only the index number of the model you want')
    quit()

# loading of deployment dataset and calculation of fingerprints
try:
    molecules_df = pd.read_csv(f'{datasets_path}/{deploy_filename}.csv')
except:
    if not os.path.exists(f'{datasets_path}/{deploy_filename}.csv'):
        print('\nFile does not exist')
    else:
        print('\nInvalid file - cannot convert to dataframe')
    quit()
if not os.path.exists(results_path):
    os.makedirs(results_path)

molecules_df = molecules_df.dropna(subset='canonical_smiles').reset_index(drop=True)
mmm.calculate_fingerprint(molecules_df, 'fingerprint/PubchemFingerprinter.xml')
fingerprint = pd.read_csv('descriptors.csv')
fingerprint = fingerprint.drop('Name', axis=1)
fingerprint = mlm.scale_features(fingerprint, preproc.MinMaxScaler())
os.remove('descriptors.csv')
os.remove('descriptors.csv.log')
os.remove('molecules.smi')
print(molecules_df.columns)
# deployment single model
re_pattern = r'(.*?)/(.*?)/(.*?).pkl'
if model_index >= 0:
    try:
        with open (model_path, 'rb') as file:
            model = pickle.load(file)
    except:
        print('Cannot read model')
    model_features = model.feature_names_in_
    feature_mask = np.isin(fingerprint.columns, model_features)
    fingerprint_aligned = fingerprint[fingerprint.columns[feature_mask]]
    y_pred = model.predict(fingerprint_aligned)
    match = re.search(re_pattern,model_path)
    y_pred = pd.Series(y_pred, name=f'pIC50_{match.group(2)}_{match.group(3)[:8]}')
    molecules_df = pd.concat([molecules_df, y_pred], axis=1)
    output_filename = mm.generate_unique_filename(
            results_path, 'deployment', match.group(2), match.group(3)[:8])

if model_index < 0:
    for model_path in model_files:
        try:
            with open (model_path, 'rb') as file:
                model = pickle.load(file)
        except:
            print('Cannot read model')
        model_features = model.feature_names_in_
        feature_mask = np.isin(fingerprint.columns, model_features)
        fingerprint_aligned = fingerprint[fingerprint.columns[feature_mask]]
        y_pred = model.predict(fingerprint_aligned)
        match = re.search(re_pattern,model_path)
        y_pred = pd.Series(y_pred, name=f'pIC50_{match.group(2)}_{match.group(3)[:8]}')
        molecules_df = pd.concat([molecules_df, y_pred], axis=1)
        output_filename = mm.generate_unique_filename(
           results_path, 'deployment', f'{len(model_files)}models')

print(molecules_df)
print(molecules_df.columns)
molecules_df.to_csv(output_filename, index=True, index_label='index')
