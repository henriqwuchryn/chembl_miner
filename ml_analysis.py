import os
import sys
import time
import re
import numpy as np
import pandas as pd
import miscelanneous_methods as mm
import machine_learning_methods as mlm
import joblib
from sklearn.ensemble import *
import sklearn.preprocessing as preproc
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import sklearn.metrics as metrics
import sklearn.model_selection as model_selection
random_state = 42

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
train_output_filename = f'{datasets_path}/{filename[:-4]}_train.csv'
test_output_filename = f'{datasets_path}/{filename[:-4]}_test.csv'
# regex to get the pre-fingerprint filename
match = re.match(
    r'^(\d+)_([a-z]+)_(\d+)_([a-z]+\-*[.0-9]+)_([a-z]+[0-9.]+)_\d+\.csv$',
    filename
)
if match == None:
    print('Invalid filename')
    quit()
activity_filename = f'{match.group(1)}_{match.group(2)}_{match.group(3)}.csv'

if not (os.path.exists(train_output_filename) and os.path.exists(test_output_filename)):
    try:
        fingerprint_df = pd.read_csv(f'{datasets_path}/{filename}', index_col='index')
    except:
        if not os.path.exists(f'{datasets_path}/{filename}'):
            print('File does not exist')
        else:
            print('Invalid file - cannot convert to dataframe')
        quit()
    fingerprint_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    fingerprint_df.dropna(inplace=True)
    features_df = fingerprint_df.drop(
        ['molecule_chembl_id','neg_log_value','bioactivity_class'],axis=1)
    target_df = fingerprint_df['neg_log_value']
    print('splitting')
    print(fingerprint_df)
    x_train, x_test, y_train, y_test = model_selection.train_test_split(
        features_df, target_df, test_size=0.2, random_state=random_state)
    train_df = pd.concat([x_train, y_train], axis=1)
    train_df.to_csv(train_output_filename, index=True, index_label='index')
    test_df = pd.concat([x_test, y_test], axis=1)
    test_df.to_csv(test_output_filename, index=True, index_label='index')
    print(f'Number of samples is: {features_df.shape[0]}')
else:
    try:
        print('reusing split')
        train_df = pd.read_csv(train_output_filename, index_col='index')
        x_train = train_df.drop('neg_log_value', axis=1)
        y_train = train_df['neg_log_value']
        test_df = pd.read_csv(test_output_filename, index_col='index')
        x_test = test_df.drop('neg_log_value', axis=1)
        y_test = test_df['neg_log_value']
        print(f'Number of samples is: {x_train.shape[0]+x_test.shape[0]}')
    except Exception as e:
        print('Invalid files')
        print(e)
        quit()

print(f'Number of features is: {x_train.shape[1]}')
print(f'size of train set is: {x_train.shape[0]}')
print(f'size of test set is: {x_test.shape[0]}')
print(f'\n{x_train.head()}')
use_scaler = input('\nDo you want to use a scaler? 1 or 0\n')
use_scaler = mm.check_if_int(use_scaler)
if use_scaler == 1:
    x_train = mlm.scale_features(x_train, preproc.StandardScaler())
    x_test = mlm.scale_features(x_test, preproc.StandardScaler())
    print(f'\nFeatures scaled\n{x_train.head()}')
    results_path = f'analysis/{filename[:-4]}/scaled'
if not os.path.exists(results_path):
    os.makedirs(results_path)

#dict structure: index, (name, algorithm)
algorithms: dict = {
    1:('AdaBoostRegressor',AdaBoostRegressor(random_state=random_state)),
    2:('BaggingRegressor',BaggingRegressor(random_state=random_state)),
    3:('ExtraTreesRegressor',ExtraTreesRegressor(random_state=random_state)),
    4:('GradientBoostingRegressor',GradientBoostingRegressor(random_state=random_state)),
    5:('HistGradientBoostingRegressor',HistGradientBoostingRegressor(random_state=random_state)),
    6:('LGBMRegressor',LGBMRegressor(random_state=random_state,verbosity=1)),
    7:('RandomForestRegressor',RandomForestRegressor(random_state=random_state)),
    8:('XGBRegressor',XGBRegressor(random_state=random_state)),
}
print('\n',pd.DataFrame([(value[0]) for key, value in algorithms.items()],columns=['Algorithm'],index=algorithms.keys()),'\n')
algorithm_index :str = input(
    'Choose which algorithm to use by inserting the index. 0 for all.\n')
algorithm_index = mm.check_if_int(algorithm_index)

if algorithm_index != 0:
    try:
        algorithm = algorithms[algorithm_index]
        print('\nAlgorithm chosen:',algorithm[0])
        optimize = input('\nDo you want to optimize the algorithms? 1 or 0\n')
        optimize = mm.check_if_int(optimize)
    except:
        print('\nIndex does not correspond to an algorithm')
        quit()
else:
    optimize = 1 #if all algorithms were selected, there is only optimization
    confirmation = input('\nAre you sure you want to use all algorithms? Only available for optimization. 1 or 0\n')
    confirmation = mm.check_if_int(confirmation)
    if confirmation == 1:
        print('\nAll algorithms will be used')
    else:
        print('\nIndex does not correspond to an algorithm')
        quit()

scoring = { #defining scoring metrics for optimization
    'r2': metrics.make_scorer(metrics.r2_score),
    'rmse': metrics.make_scorer(
        lambda y_true, y_pred: metrics.root_mean_squared_error(y_true, y_pred)),
    'mae': metrics.make_scorer(metrics.mean_absolute_error)
}

if optimize == 1:
    #dict structure: name, params
    param_grids = {
        'BaggingRegressor': {
            'n_estimators': [10, 20, 40],#10
            'max_samples': [0.7, 1.0],#1.0
            'max_features': [0.7, 1.0],#1.0
            'bootstrap': [True, False],  # Whether samples are drawn with replacement
            'bootstrap_features': [True, False]  # Whether features are drawn with replacement
        },
        'GradientBoostingRegressor': {
            'n_estimators': [1200, 1600, 2000], #100
            'learning_rate': [0.005, 0.01, 0.015], #0.1
            'max_depth': [21], #3
            'min_samples_split': [2], #2
            'min_samples_leaf': [4], #1
            'subsample': [0.3, 0.5, 0.7],  # 1.0
            'max_features': ['sqrt']  # 1.0
        },
        'LGBMRegressor': {
            'n_estimators': [100, 200, 400], #100
            'learning_rate': [0.05, 0.1, 0.2],#0.1
            'max_depth': [6, 9, 18],#-1
            'num_leaves': [31, 62],#31
            'min_child_samples': [20, 40],#20
            'subsample': [0.7, 1.0],  # Fraction of samples used for fitting
            'colsample_bytree': [0.7, 1.0],  # Fraction of features used for fitting
            'reg_alpha': [0, 0.1, 1],  # L1 regularization
            'reg_lambda': [0, 0.1, 1],  # L2 regularization
            'force_row_wise': [True]
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
            'n_estimators': [600,800,1200],
            'learning_rate': [0.01,0.025],
            'max_depth': [9],
            'min_child_weight': [0.5,4],  # Minimum sum of instance weight needed in a child
            'gamma': [0.5],  # Minimum loss reduction required to make a split
            'subsample': [0.5, 0.7],  # Fraction of samples used for fitting
            'colsample_bytree': [0.5,0.7],  # Fraction of features used for fitting
            'reg_alpha': [1],  # L1 regularization
            'reg_lambda': [1]  # L2 regularization
        },
        'HistGradientBoostingRegressor': {
            'loss': ['squared_error'], #squared_error
            'max_iter': [1600, 2000], #100
            'learning_rate': [0.01, 0.025], #0.1
            'max_depth': [None], #None
            'min_samples_leaf': [40, 80, 160], #20
            'max_leaf_nodes': [62, 124],  #61
            'l2_regularization': [0, 0.5, 1],  #0
            'max_bins': [125, 255]  #255
        },
        'ExtraTreesRegressor': {
            'n_estimators': [100, 200, 300], #100
            'max_depth': [6, 9, None], #None
            'min_samples_split': [2, 4], #2
            'min_samples_leaf': [1, 2], #1
            'max_features': [1.0, 'sqrt', 'log2'],  # Number of features to consider for splits
            'bootstrap': [True, False],  # Whether bootstrap samples are used
            # 'oob_score': [True, False]  # Whether to use out-of-bag samples for estimation
        },
        'AdaBoostRegressor': {
            'n_estimators': [50, 100, 200], #50
            'learning_rate': [0.5, 1.0, 2.0], #1.0
            'loss': ['linear', 'square', 'exponential']  #linear
        }
    }
    if algorithm_index == 0:
        for index, (name, alg) in algorithms.items():
            search_cv_results, best_params, time_to_execute = mlm.evaluate_and_optimize(
                alg, param_grids[name], x_train, y_train, scoring, name)
            search_output_filename = mm.generate_unique_filename(
                results_path, name, 'gridsearch')
            search_cv_results.to_csv(search_output_filename)
            search_txtoutput_filename = mm.generate_unique_filename(
                results_path, name, 'bestparams', 'time', suffix='.txt')
            with open(search_txtoutput_filename, 'w') as file:
                file.write(f'Evaluated parameters: {str(param_grids[name])}\nBest parameters: {str(best_params)}\nTime to run: {time_to_execute}\n') # writing parameters to text file
        print(f"\nOptimization completed. Results saved to {search_output_filename}.")
        quit() # following code only supports one algorithm at a time

    else:
        search_cv_results, best_params, time_to_execute = mlm.evaluate_and_optimize(
            algorithm[1], param_grids[algorithm[0]], x_train, y_train, scoring, algorithm[0])
        search_output_filename = mm.generate_unique_filename(
            results_path, algorithm[0], 'gridsearch')
        search_cv_results.to_csv(search_output_filename)
        search_txtoutput_filename = mm.generate_unique_filename(
            results_path, algorithm[0], 'bestparams', 'time', suffix='.txt')
        with open(search_txtoutput_filename, 'w') as file:
            file.write(f'Evaluated parameters: {str(param_grids[algorithm[0]])}\nBest parameters: {str(best_params)}\nTime to run: {time_to_execute}\n') # writing parameters to text file
        print(f"\nOptimization completed. Results saved to {search_output_filename}, {search_txtoutput_filename}.")

try:
    params = best_params
except: #if above fails, ask for input
    params = input(f'Insert the best parameters for the algorithm {algorithm[0]}\n')
    try:
        params = dict(eval(params))
    except:
        print('\nInvalid parameters')
        quit()

optimized_algorithm = algorithm[1].set_params(**params)
print('\nPerforming supervised outlier removal')

x_train_clean, y_train_clean, cv_results = mlm.supervised_outlier_removal(
    algorithm=optimized_algorithm, x_train=x_train, y_train= y_train,
    scoring=scoring, algorithm_name=algorithm[0])
print('Number of samples before cleaning:', x_train.shape[0])
print('Number of samples after cleaning:', x_train_clean.shape[0])
print(f'Removed {round(((x_train.shape[0]-x_train_clean.shape[0])/x_train.shape[0]*100),2)}% of samples')

activity_df = pd.read_csv(f'{datasets_path}/{activity_filename}', index_col='index')
full_df = pd.concat([activity_df,x_train], axis=1)
outlier_mask = np.logical_not(full_df.index.isin(x_train_clean.index))
outlier_df = full_df[outlier_mask]
outlier_output_filename = mm.generate_unique_filename(
    datasets_path, filename[:-4], algorithm[0], 'outliers')
outlier_df.to_csv(outlier_output_filename, index=True, index_label='index')
print(f'\nOutliers are available at {outlier_output_filename}')

r2_cv = cv_results['test_r2'].mean()
rmse_cv = cv_results['test_rmse'].mean()
mae_cv = cv_results['test_mae'].mean()
score_df_cv = pd.DataFrame(
    {'r2':r2_cv, 'rmse':rmse_cv, 'mae':mae_cv}, index=['score_cv'])
r2_train = cv_results['train_r2'].mean()
rmse_train = cv_results['train_rmse'].mean()
mae_train = cv_results['train_mae'].mean()
score_df_train = pd.DataFrame(
    {'r2':r2_train, 'rmse':rmse_train, 'mae':mae_train}, index=['score_train'])

cv_results_clean = model_selection.cross_validate(
    estimator=optimized_algorithm, X=x_train_clean, y=y_train_clean, cv=10,
    scoring=scoring, return_train_score=True)
r2_cv_clean = cv_results_clean['test_r2'].mean()
rmse_cv_clean = cv_results_clean['test_rmse'].mean()
mae_cv_clean = cv_results_clean['test_mae'].mean()
score_df_cv_clean = pd.DataFrame(
    {'r2':r2_cv_clean, 'rmse':rmse_cv_clean, 'mae':mae_cv_clean}, index=['score_cv_clean'])
r2_train_clean = cv_results_clean['train_r2'].mean()
rmse_train_clean = cv_results_clean['train_rmse'].mean()
mae_train_clean = cv_results_clean['train_mae'].mean()
score_df_train_clean = pd.DataFrame(
    {'r2':r2_train_clean, 'rmse':rmse_train_clean, 'mae':mae_train_clean}, index=['score_train_clean'])

score_df_final = pd.concat([score_df_cv, score_df_train, score_df_cv_clean, score_df_train_clean], axis=0)
print(score_df_final)
score_output_filename = mm.generate_unique_filename(
    results_path, algorithm[0], 'scores')
score_df_final.to_csv(score_output_filename)

# trained models output removed, as it is not the point of this analysis

# model_output_filename = mm.generate_unique_filename(
#     results_path, algorithm[0], 'Model',suffix='.pkl')
# joblib.dump(model, model_output_filename)
# model_clean_output_filename = mm.generate_unique_filename(
#     results_path, algorithm[0], 'ModelClean',suffix='.pkl')
# joblib.dump(model_clean, model_clean_output_filename)

print(f'\nResults are available at {score_output_filename}')
