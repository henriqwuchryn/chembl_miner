import os
import sys
import time
import re
import numpy as np
import pandas as pd
import miscelanneous_methods as mm
import machine_learning_methods as mlm
from dataset_wrapper import DatasetWrapper
from sklearn_genetic.space import Categorical, Integer, Continuous
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
if not os.path.exists(results_path):
    os.makedirs(results_path)
datasets_folder = 'datasets'
datasets_path = f'{datasets_folder}/{filename[:-4]}'
parameter_path = 'config/parameter_grids'

general_columns = ["canonical_smiles","molecule_chembl_id","standard_type","standard_value","standard_units","assay_description","MW","LogP","NumHDonors","NumHAcceptors","Ro5Violations","bioactivity_class"]
target_column = 'neg_log_value'

#Available algorithms
#dict structure: index, (name, algorithm)
algorithms: dict = {
    1:('BaggingRegressor',BaggingRegressor(random_state=random_state)),
    2:('ExtraTreesRegressor',ExtraTreesRegressor(random_state=random_state)),
    3:('GradientBoostingRegressor',GradientBoostingRegressor(random_state=random_state)),
    4:('HistGradientBoostingRegressor',HistGradientBoostingRegressor(random_state=random_state)),
    5:('LGBMRegressor',LGBMRegressor(random_state=random_state,verbosity=1)),
    6:('RandomForestRegressor',RandomForestRegressor(random_state=random_state)),
    7:('XGBRegressor',XGBRegressor(random_state=random_state)),
}
print(f"\n{pd.DataFrame([(value[0]) for key, value in algorithms.items()],columns=['Algorithm'],index=algorithms.keys())}\n")
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
        print('\nQuitting')
        quit()

scoring = { #defining scoring metrics for optimization
    'r2': metrics.make_scorer(metrics.r2_score),
    'rmse': metrics.make_scorer(
        lambda y_true, y_pred: metrics.root_mean_squared_error(y_true, y_pred)),
    'mae': metrics.make_scorer(metrics.mean_absolute_error)
}

if not os.path.exists(f'{datasets_path}/gd.csv'):
    data = DatasetWrapper().load_raw_dataset(
            f'{datasets_folder}/{filename}',
            general_columns,
            target_column)
    data.save(datasets_path)
else:
    data = DatasetWrapper().load_dataset(datasets_path)

data.describe()

if optimize == 1:
    #dict structure: name, params
    if not os.path.exists(parameter_path):
        print('\nNo parameter_grids file on config folder')
        quit()
    with open(parameter_path,'r') as file:
        param_grids = eval(file.read())

    if algorithm_index == 0:
        for index, (name, alg) in algorithms.items():
            try:
                param_grids[name]
            except:
                print('no parameter grid for chosen algorithm')
            search_cv_results, best_params, time_to_execute = mlm.evaluate_and_optimize(
                alg, param_grids[name], data.x_preprocessing, data.y_preprocessing, scoring, name)
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
            algorithm[1], param_grids[algorithm[0]], data.x_preprocessing, data.y_preprocessing, scoring, algorithm[0])
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
    algorithm=optimized_algorithm, x_train=data.x_train, y_train= data.y_train,
    scoring=scoring, algorithm_name=algorithm[0])
print('Number of samples before cleaning:', data.x_train.shape[0])
print('Number of samples after cleaning:', x_train_clean.shape[0])
print(f'Removed {round(((data.x_train.shape[0]-x_train_clean.shape[0])/data.x_train.shape[0]*100),2)}% of samples')
outlier_mask = np.logical_not(data.general_data.index.isin(x_train_clean.index))
outlier_df = data.general_data[outlier_mask]
outlier_output_filename = mm.generate_unique_filename(
    datasets_folder, filename[:-4], algorithm[0], 'outliers')
outlier_df.to_csv(outlier_output_filename, index=True, index_label='index')
print(f'\nOutliers are available at {outlier_output_filename}')

#assembling cv_results dataframe
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

print('Evaluating model with cleaned data')
cv_results_clean = model_selection.cross_validate(
    estimator=optimized_algorithm, X=x_train_clean, y=y_train_clean, cv=10,
    scoring=scoring, return_train_score=True)
#assembling cv_results_clean dataframe
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
