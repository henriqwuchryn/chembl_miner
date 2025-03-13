import pandas as pd
import seaborn as sns
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import joblib
from sklearn.ensemble import *
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import sklearn.metrics as metrics
import sklearn.model_selection as model_selection
import sklearn.preprocessing as preprocessing
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

fingerprint_df.replace([np.inf, -np.inf], np.nan, inplace=True)
fingerprint_df.dropna(inplace=True)
fingerprint_df.reset_index(drop=True, inplace=True)
features_df = fingerprint_df.drop(['molecule_chembl_id',
                                    'neg_log_value',
                                    'bioactivity_class'],
                                    axis=1)
target_df = fingerprint_df['neg_log_value']
x_train, x_test, y_train, y_test = model_selection.train_test_split(features_df,
                                                    target_df,
                                                    test_size=0.2,
                                                    random_state=random_state)
print(f'Number of features is: {features_df.shape[1]}')
print(f'Number of samples is: {features_df.shape[0]}')
print(f'size of train set is: {x_train.shape[0]}')
print(f'size of test set is: {x_test.shape[0]}')
algorithms: dict = {1:('BaggingRegressor',BaggingRegressor(random_state=random_state)),
                    2:('GradientBoostingRegressor',GradientBoostingRegressor(random_state=random_state)),
                    3:('LGBMRegressor',LGBMRegressor(random_state=random_state)),
                    4:('RandomForestRegressor',RandomForestRegressor(random_state=random_state)),
                    5:('XGBRegressor',XGBRegressor(random_state=random_state)),
                    6:('HistGradientBoostingRegressor',HistGradientBoostingRegressor(random_state=random_state)),
                    7:('ExtraTreesRegressor',ExtraTreesRegressor(random_state=random_state)),
                    8:('AdaBoostRegressor',AdaBoostRegressor(random_state=random_state))
                    }
print('\n',pd.DataFrame([(key, value[0]) for key, value in algorithms.items()],columns=['Index','Algorithm']),'\n')
algorithm_index :str = input('Choose which algorithm to use by inserting the index. 0 for all.\n')
algorithm_index = mmm.check_if_int(algorithm_index)

if algorithm_index != 0:
    try:
        algorithm = algorithms[algorithm_index]
        print('\nAlgorithm chosen:',algorithm)
    except:
        print('\nIndex does not correspond to an algorithm')
        quit()
else:
    confirmation = input('\nAre you sure you want to use all algorithms? 1 or 0\n')
    confirmation = mmm.check_if_int(confirmation)
    if confirmation == 1:
        print('\nAll algorithms will be used')
    else:
        print('\nIndex does not correspond to an algorithm')
        quit()         

scoring = {
    'r2': metrics.make_scorer(metrics.r2_score),
    'rmse': metrics.make_scorer(lambda y_true, y_pred: metrics.mean_squared_error(y_true, y_pred)),
    'mae': metrics.make_scorer(metrics.mean_absolute_error)
}
#dict structure: name, param_grid
param_grids = {
    'BaggingRegressor': {
        'n_estimators': [10, 20, 40],#10
        'max_samples': [0.7, 1.0],#1.0
        'max_features': [0.7, 1.0],#1.0
        'bootstrap': [True, False],  # Whether samples are drawn with replacement
        'bootstrap_features': [True, False]  # Whether features are drawn with replacement
    },
    'GradientBoostingRegressor': {
        'n_estimators': [100, 200, 400], #100
        'learning_rate': [0.05, 0.1, 0.2], #0.1
        'max_depth': [3, 5, 7], #3
        'min_samples_split': [2, 4], #2
        'min_samples_leaf': [1, 2], #1
        'subsample': [0.7, 1.0],  # 1.0
        'max_features': [1.0, 'sqrt', 'log2']  # 1.0
    },
    'LGBMRegressor': {
        'n_estimators': [100, 200, 400], #100
        'learning_rate': [0.05, 0.1, 0.2],#0.1
        'max_depth': [-1, 6, 9],#-1
        'num_leaves': [31, 62],#62
        'min_child_samples': [20, 40],#40
        'subsample': [0.7, 1.0],  # Fraction of samples used for fitting
        'colsample_bytree': [0.7, 1.0],  # Fraction of features used for fitting
        'reg_alpha': [0, 0.1],  # L1 regularization
        'reg_lambda': [0, 0.1]  # L2 regularization
    },
    'RandomForestRegressor': {
        'n_estimators': [100, 200, 400], #100
        'max_depth': [6, 9, None], #None
        'min_samples_split': [2, 4], #2
        'min_samples_leaf': [1, 2], #1
        'max_features': [1.0, 'sqrt', 'log2'],  # 1.0
        'bootstrap': [True, False],  # Whether bootstrap samples are used
        'oob_score': [True, False]  # Whether to use out-of-bag samples for estimation
    },
    'XGBRegressor': {
        'n_estimators': [50, 100, 200, 300],
        'learning_rate': [0.15, 0.3, 0.6],
        'max_depth': [3, 6, 9],
        'min_child_weight': [1, 2],  # Minimum sum of instance weight needed in a child
        'gamma': [0, 0.1],  # Minimum loss reduction required to make a split
        'subsample': [0.7, 1.0],  # Fraction of samples used for fitting
        'colsample_bytree': [0.7, 1.0],  # Fraction of features used for fitting
        'reg_alpha': [0, 0.1],  # L1 regularization
        'reg_lambda': [0, 0.1]  # L2 regularization
    },
    'HistGradientBoostingRegressor': {
        'max_iter': [100, 200, 400], #100
        'learning_rate': [0.05, 0.1, 0.2], #0.1
        'max_depth': [6, 9, None], #None
        'min_samples_leaf': [20, 40], #20
        'max_leaf_nodes': [31, 62],  #61
        'l2_regularization': [0, 0.1],  #0
        'max_bins': [255, 610]  #255
    },
    'ExtraTreesRegressor': {
        'n_estimators': [100, 200, 300], #100
        'max_depth': [6, 9, None], #None
        'min_samples_split': [2, 4], #2
        'min_samples_leaf': [1, 2], #1
        'max_features': [1.0, 'sqrt', 'log2'],  # Number of features to consider for splits
        'bootstrap': [True, False],  # Whether bootstrap samples are used
        'oob_score': [True, False]  # Whether to use out-of-bag samples for estimation
    },
    'AdaBoostRegressor': {
        'n_estimators': [50, 100, 200], #50
        'learning_rate': [0.5, 1.0, 2.0], #1.0
        'loss': ['linear', 'square', 'exponential']  #linear
    }
}

print(target_df.describe())


def evaluate_and_optimize(algorithm, param_grid, X_train, y_train,algorithm_name):
    print(f"\nOptimizing {algorithm_name}")
    print(f"Parameters: {param_grid}")
    grid_search = model_selection.GridSearchCV(estimator=algorithm,
                                            param_grid=param_grid,
                                            scoring=scoring,
                                            refit='r2',
                                            n_jobs=-1)
    print('\nFitting\n')
    grid_search.fit(X_train, y_train)                                   
    results = pd.DataFrame(grid_search.cv_results_)
    print(f'Results:\n{results}')
    print(f'\nBest parameters:\n{grid_search.best_params_}')
    print(f'\nBest score index:\n{grid_search.best_index_}')
    results.to_csv(f'{results_path}/{algorithm_name}_GridSearch.csv', index=False)
    return results


if algorithm_index == 0 and confirmation == 1:
    for index, (name, algorithm) in algorithms.items():
        evaluate_and_optimize(algorithm, param_grids[name], x_train, y_train, name)
else:
    name, algorithm = algorithms[algorithm_index]
    evaluate_and_optimize(algorithm, param_grids[name], x_train, y_train, name)

print(f"\nOptimization completed. Results saved to {results_path}.")