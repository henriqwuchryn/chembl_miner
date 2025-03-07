import pandas as pd
import seaborn as sns
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import joblib
from pycaret import regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import cross_validate, cross_val_score
random_state = 42
sns.set_theme(style='whitegrid',font='liberation serif')
import molecules_manipulation_methods as mmm
import os
import sys

try:
    filename = sys.argv[1]
except:
    print(
        '''\nyou must insert the dataset filename as an argument, like this:
    >python ml_analysis.py FILENAME.csv'''
    )
    quit()

results_path = f'analysis/{filename[:-4]}'
datasets_path = 'datasets'

try:
    fingerprint_df = pd.read_csv(f'{datasets_path}/{filename}')
except:
    if not os.path.exists(f'{datasets_path}/{filename}'):
        print('File does not exist')
    else:
        print('Invalid file - cannot convert to dataframe')
    quit()
if not os.path.exists(results_path):
    os.makedirs(results_path)

features_df = fingerprint_df.drop(['molecule_chembl_id',
                                    'neg_log_value',
                                    'bioactivity_class'],
                                    axis=1)
target_df = fingerprint_df['neg_log_value']
data_df = pd.concat([features_df,target_df],axis=1)
compare = input('\nDo you want to compare models? 1 or 0\n')
compare = mmm.check_if_int(compare)

if compare == 1:
    regression_setup = regression.setup(data=data_df,
                                        target='neg_log_value',
                                        train_size=0.8)
    compare_models = regression.compare_models(turbo=True,fold=10)
    print(regression.get_metrics())
    

algorithms: dict = {'01':BaggingRegressor(),
                    '02':GradientBoostingRegressor(),
                    '03':LGBMRegressor(),
                    '04':RandomForestRegressor(),
                    '05':XGBRegressor()}
print('\n',pd.DataFrame(algorithms.items(),columns=['Index','Algorithm']),'\n')
algorithm_index :str = input('Choose which algorithm to use by inserting the index\n')

try:
    algorithm_index = int(algorithm_index)
except:
    print('\nPlease type only the index number of the algorithm you want')
    quit()
try:
    algorithm = list(algorithms.values())[algorithm_index]
except:
    print('\nIndex unavailable')

x_train, x_test, y_train, y_test = train_test_split(features_df,
                                                    target_df,
                                                    test_size=0.2,
                                                    random_state=random_state)
print(f'Number of features is: {features_df.shape[1]}')
print(f'Number of samples is: {features_df.shape[0]}')
print(f'size of train set is: {x_train.shape[0]}')
print(f'size of test set is: {x_test.shape[0]}')
