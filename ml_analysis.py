import os
import sys
import time
import numpy as np
import pandas as pd
import miscelanneous_methods as mm
import machine_learning_methods as mlm
import joblib
from sklearn.ensemble import *
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
features_df = fingerprint_df.drop(
    ['molecule_chembl_id','neg_log_value','bioactivity_class'],axis=1)
target_df = fingerprint_df['neg_log_value']
x_train, x_test, y_train, y_test = model_selection.train_test_split(
    features_df, target_df, test_size=0.2, random_state=random_state)
print(f'Number of features is: {features_df.shape[1]}')
print(f'Number of samples is: {features_df.shape[0]}')
print(f'size of train set is: {x_train.shape[0]}')
print(f'size of test set is: {x_test.shape[0]}')
#dict structure: index, (name, algorithm)
algorithms: dict = {
    1:('BaggingRegressor',BaggingRegressor(random_state=random_state)),
    2:('GradientBoostingRegressor',GradientBoostingRegressor(random_state=random_state)),
    3:('LGBMRegressor',LGBMRegressor(random_state=random_state,verbosity=-1)),
    4:('RandomForestRegressor',RandomForestRegressor(random_state=random_state)),
    5:('XGBRegressor',XGBRegressor(random_state=random_state)),
    6:('HistGradientBoostingRegressor',HistGradientBoostingRegressor(random_state=random_state)),
    7:('ExtraTreesRegressor',ExtraTreesRegressor(random_state=random_state)),
    8:('AdaBoostRegressor',AdaBoostRegressor(random_state=random_state))
}
print('\n',pd.DataFrame([(key, value[0]) for key, value in algorithms.items()],columns=['Index','Algorithm']),'\n')
algorithm_index :str = input(
    'Choose which algorithm to use by inserting the index. 0 for all.\n')
algorithm_index = mm.check_if_int(algorithm_index)

if algorithm_index != 0:
    try:
        algorithm = algorithms[algorithm_index]
        print('\nAlgorithm chosen:',algorithm[0])
    except:
        print('\nIndex does not correspond to an algorithm')
        quit()
else:
    confirmation = input('\nAre you sure you want to use all algorithms? Only available for optimization. 1 or 0\n')
    confirmation = mm.check_if_int(confirmation)
    if confirmation == 1:
        print('\nAll algorithms will be used')
    else:
        print('\nIndex does not correspond to an algorithm')
        quit()

scoring = {
    'r2': metrics.make_scorer(metrics.r2_score),
    'rmse': metrics.make_scorer(
        lambda y_true, y_pred: metrics.root_mean_squared_error(y_true, y_pred)),
    'mae': metrics.make_scorer(metrics.mean_absolute_error)
}

optimize = input('Do you want to optimize the algorithms? 1 or 0\n')
optimize = mm.check_if_int(optimize)
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
    if algorithm_index == 0 and confirmation == 1:
        for index, (name, alg) in algorithms.items():
            search_cv_results, best_params = mlm.evaluate_and_optimize(
                alg, param_grids[name], x_train, y_train, scoring, name)
            search_output_filename = mm.generate_unique_filename(
                results_path, name, 'GridSearch')
            search_cv_results.to_csv(search_output_filename, index=False)
    else:
        search_cv_results, best_params = mlm.evaluate_and_optimize(
            algorithm[1], param_grids[algorithm[0]], x_train, y_train, scoring, algorithm[0])
        search_output_filename = mm.generate_unique_filename(
            results_path, algorithm[0], 'GridSearch')
        search_cv_results.to_csv(search_output_filename, index=False)


    print(f"\nOptimization completed. Results saved to {results_path}.")

try:
    params = best_params
except:
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
r2_cv = cv_results['test_r2'].mean()
rmse_cv = cv_results['test_rmse'].mean()
mae_cv = cv_results['test_mae'].mean()
score_df_cv = pd.DataFrame(
    {'r2':r2_cv, 'rmse':rmse_cv, 'mae':mae_cv}, index=['score_cv'])

model = optimized_algorithm
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
r2 = metrics.r2_score(y_test, y_pred)
rmse = metrics.root_mean_squared_error(y_test, y_pred)
mae = metrics.mean_absolute_error(y_test, y_pred)
score_df = pd.DataFrame(
    {'r2':r2, 'rmse':rmse, 'mae':mae}, index=['score'])

model_clean = optimized_algorithm
model_clean.fit(x_train_clean, y_train_clean)
y_pred_clean = model_clean.predict(x_test)
r2_clean = metrics.r2_score(y_test, y_pred_clean)
rmse_clean = metrics.root_mean_squared_error(y_test, y_pred_clean)
mae_clean = metrics.mean_absolute_error(y_test, y_pred_clean)
score_df_clean = pd.DataFrame(
    {'r2':r2_clean, 'rmse':rmse_clean, 'mae':mae_clean}, index=['score_clean'])

cv_results_clean = model_selection.cross_validate(
    estimator=model_clean, X=x_train_clean, y=y_train_clean, cv=10,
    scoring=scoring, return_estimator=True, return_indices=True)
r2_cv_clean = cv_results['test_r2'].mean()
rmse_cv_clean = cv_results['test_rmse'].mean()
mae_cv_clean = cv_results['test_mae'].mean()
score_df_cv_clean = pd.DataFrame(
    {'r2':r2_cv_clean, 'rmse':rmse_cv_clean, 'mae':mae_cv_clean}, index=['score_cv_clean'])

score_df_final = pd.concat([score_df_cv, score_df, score_df_cv_clean, score_df_clean], axis=0)
print(score_df_final)
score_output_filename = mm.generate_unique_filename(
    results_path, algorithm[0], 'Scores')
score_df_final.to_csv(score_output_filename)
model_output_filename = mm.generate_unique_filename(
    results_path, algorithm[0], 'Model',suffix='.pkl')
joblib.dump(model, model_output_filename)
model_clean_output_filename = mm.generate_unique_filename(
    results_path, algorithm[0], 'ModelClean',suffix='.pkl')
print(f'\nResults are available at {results_path}')
