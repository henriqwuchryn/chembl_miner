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

#PYCARET COMPARE MODELS WAS NOT WORKING PROPERLY, WON'T BE USING IT

# compare = input('\nDo you want to compare models? 1 or 0\n')
# compare = mmm.check_if_int(compare)

# if compare == 1:
#     data_df = pd.concat([features_df,target_df],axis=1)
#     regression_setup = regression.setup(data=data_df,
#                                         target='neg_log_value',
#                                         train_size=0.8)
#     compare_models = regression.compare_models(turbo=True,fold=10)
#     print(regression.get_metrics())
    

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
        print('\nIndex unavailable')
else:
    print('\nAll algorithms will be used')


scoring = {
    'r2': metrics.make_scorer(metrics.r2_score),
    'rmse': metrics.make_scorer(lambda y_true, y_pred: metrics.mean_squared_error(y_true, y_pred)),
    'mae': metrics.make_scorer(metrics.mean_absolute_error)
}
param_grids = {
    'BaggingRegressor': {
        'n_estimators': [10, 50, 100, 200],
        'max_samples': [0.5, 0.8, 1.0],
        'max_features': [0.5, 0.8, 1.0],
        'bootstrap': [True, False],  # Whether samples are drawn with replacement
        'bootstrap_features': [True, False]  # Whether features are drawn with replacement
    },
    'GradientBoostingRegressor': {
        'n_estimators': [50, 100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7, 9],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'subsample': [0.8, 0.9, 1.0],  # Fraction of samples used for fitting
        'max_features': ['auto', 'sqrt', 'log2']  # Number of features to consider for splits
    },
    'LGBMRegressor': {
        'n_estimators': [50, 100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7, 9],
        'num_leaves': [31, 50, 100],  # Maximum number of leaves in one tree
        'min_child_samples': [20, 50, 100],  # Minimum number of data in one leaf
        'subsample': [0.8, 0.9, 1.0],  # Fraction of samples used for fitting
        'colsample_bytree': [0.8, 0.9, 1.0],  # Fraction of features used for fitting
        'reg_alpha': [0, 0.1, 1],  # L1 regularization
        'reg_lambda': [0, 0.1, 1]  # L2 regularization
    },
    'RandomForestRegressor': {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [3, 5, 7, 9, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt', 'log2'],  # Number of features to consider for splits
        'bootstrap': [True, False],  # Whether bootstrap samples are used
        'oob_score': [True, False]  # Whether to use out-of-bag samples for estimation
    },
    'XGBRegressor': {
        'n_estimators': [50, 100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7, 9],
        'min_child_weight': [1, 5, 10],  # Minimum sum of instance weight needed in a child
        'gamma': [0, 0.1, 0.2],  # Minimum loss reduction required to make a split
        'subsample': [0.8, 0.9, 1.0],  # Fraction of samples used for fitting
        'colsample_bytree': [0.8, 0.9, 1.0],  # Fraction of features used for fitting
        'reg_alpha': [0, 0.1, 1],  # L1 regularization
        'reg_lambda': [0, 0.1, 1]  # L2 regularization
    },
    'HistGradientBoostingRegressor': {
        'max_iter': [50, 100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7, 9],
        'min_samples_leaf': [1, 2, 4, 10],
        'max_leaf_nodes': [31, 50, 100],  # Maximum number of leaves
        'l2_regularization': [0, 0.1, 1],  # L2 regularization
        'max_bins': [128, 256, 512]  # Maximum number of bins for feature discretization
    },
    'ExtraTreesRegressor': {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [3, 5, 7, 9, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt', 'log2'],  # Number of features to consider for splits
        'bootstrap': [True, False],  # Whether bootstrap samples are used
        'oob_score': [True, False]  # Whether to use out-of-bag samples for estimation
    },
    'AdaBoostRegressor': {
        'n_estimators': [50, 100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'loss': ['linear', 'square', 'exponential']  # Loss function to optimize
    }
}

print(target_df.describe())

##BELOW IS AI GENERATED, IN REVIEW

def evaluate_and_optimize(model, param_grid, X_train, y_train, X_test, y_test, algorithm_name):
    print(f"\nOptimizing {algorithm_name}...")
    
    # Use GridSearchCV for parameter optimization
    grid_search = model_selection.GridSearchCV(estimator=model, param_grid=param_grid)
    grid_search.fit(X_train, y_train)
    print(grid_search.cv_results_)
    
    # Get the best model
    best_model = grid_search.best_estimator_
    
    # Evaluate on test set
    y_pred = best_model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    
    # Save results
    results = {
        'Algorithm': algorithm_name,
        'Best Parameters': grid_search.best_params_,
        'Test R2': r2,
        'Test RMSE': rmse,
        'Test MAE': mae
    }
    
    # Save model to file
    model_filename = f'{results_path}/{algorithm_name}_best_model.pkl'
    joblib.dump(best_model, model_filename)
    print(f"Best model saved to {model_filename}")
    
    return results

# Evaluate selected algorithm(s)
results_list = []

if algorithm_index == 0:
    # Evaluate all algorithms
    for idx, (name, model) in algorithms.items():
        results = evaluate_and_optimize(model, param_grids[name], x_train, y_train, x_test, y_test, name)
        results_list.append(results)
else:
    # Evaluate the selected algorithm
    if algorithm_index in algorithms:
        name, model = algorithms[algorithm_index]
        results = evaluate_and_optimize(model, param_grids[name], x_train, y_train, x_test, y_test, name)
        results_list.append(results)
    else:
        print('\nIndex unavailable')

# Save results to CSV
results_df = pd.DataFrame(results_list)
# results_filename = f'{results_path}/optimization_results.csv'
# results_df.to_csv(results_filename, index=False)

print(f"\nOptimization completed. Results saved to {results_filename}.")
print(results_df)