import os
import pickle
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
    if filename[-4:] == '.csv':
        filename = filename[:-4]
except:
    print(
        '''\nyou must insert the dataset filename as an argument, like this:
>python ml_analysis.py FILENAME'''
    )
    quit()

results_path = f'analysis/{filename}'
if not os.path.exists(results_path):
    os.makedirs(results_path)
datasets_folder = 'datasets'
datasets_path = f'{datasets_folder}/{filename}'
parameter_path = 'config/parameter_grids'
config_path = 'config/config'

general_columns = ["canonical_smiles","molecule_chembl_id","standard_type","standard_value","standard_units","assay_description","MW","LogP","NumHDonors","NumHAcceptors","Ro5Violations","bioactivity_class"]
target_column = 'neg_log_value'

#Available algorithms
#dict structure: index, (name, algorithm)
algorithms: dict = {
    1:('BaggingRegressor',BaggingRegressor(random_state=random_state)),
    2:('ExtraTreesRegressor',ExtraTreesRegressor(random_state=random_state)),
    3:('GradientBoostingRegressor',GradientBoostingRegressor(random_state=random_state)),
    4:('HistGradientBoostingRegressor',HistGradientBoostingRegressor(random_state=random_state)),
    5:('RandomForestRegressor',RandomForestRegressor(random_state=random_state)),
    6:('XGBRegressor',XGBRegressor(random_state=random_state)),
}
print(f"\n{pd.DataFrame([(value[0]) for key, value in algorithms.items()],columns=['Algorithm'],index=algorithms.keys())}\n")
algorithm_index :str = input(
    'Choose which algorithm to use by inserting the index. 0 for all.\n')
algorithm_index = mm.check_if_int(algorithm_index)

if algorithm_index != 0:
    try:
        alg = algorithms[algorithm_index][1]
        name = algorithms[algorithm_index][0]
        print('\nAlgorithm chosen:',name)
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
            f'{datasets_folder}/{filename}.csv',
            general_columns,
            target_column)
    data.save(datasets_path)
else:
    data = DatasetWrapper().load_dataset(datasets_path)

data.describe()

try:
    with open(config_path,'r') as file:
        config = eval(file.read())
        gen = config['generations']
        pop = config['population_size']
except:
    gen = 30
    pop = 30

if optimize == 1:
    #dict structure: name, params
    if not os.path.exists(parameter_path):
        print('\nNo parameter_grids file on config folder')
        quit()
    with open(parameter_path,'r') as file:
        params_dict = eval(file.read())


    if algorithm_index == 0:
        for index, (name, alg) in algorithms.items():
            try:
                params = params_dict[name]
            except Exception as e:
                print(e)
                print(f'\nNo parameters for {name} in {parameter_path}')
                continue

            search_cv_results, best_params, time = mlm.evaluate_and_optimize(
                algorithm=alg,
                param_grid=params,
                x_train=data.x_preprocessing,
                y_train=data.y_preprocessing,
                scoring=scoring,
                algorithm_name=name,
                generations=gen,
                population_size=pop)
            search_output_filename = mm.generate_unique_filename(
                results_path, name, 'paramsearch')
            search_txtoutput_filename = mm.generate_unique_filename(
                results_path, name, 'bestparams', 'time', suffix='.txt')
            search_cv_results.to_csv(search_output_filename)
            with open(search_txtoutput_filename, 'w') as file:
                file.write(f'Evaluated parameters: {mlm.describe_params(params)}\nBest parameters: {str(best_params)}\nTime to run: {time}\n') # writing parameters to text file
        print(f"\nOptimization completed. Results saved to {search_output_filename}.")
        quit() # following code only supports one algorithm at a time

    else:
        params = params_dict[name]
        try:
            params = params_dict[name]
        except Exception as e:
            print(e)
            print(f'\nNo parameters for {name} in {parameter_path}')
            print('Quitting')
            quit()
        search_cv_results, best_params, time = mlm.evaluate_and_optimize(
            algorithm=alg,
            param_grid=params,
            x_train=data.x_preprocessing,
            y_train=data.y_preprocessing,
            scoring=scoring,
            algorithm_name=name,
            generations=gen,
            population_size=pop)
        search_output_filename = mm.generate_unique_filename(
            results_path, name, 'paramsearch')
        search_txtoutput_filename = mm.generate_unique_filename(
            results_path, name, 'bestparams', 'time', suffix='.txt')
        search_cv_results.to_csv(search_output_filename)
        with open(search_txtoutput_filename, 'w') as file:
            file.write(f'Evaluated parameters: {mlm.describe_params(params)}\nBest parameters: {str(best_params)}\nTime to run: {time}\n') # writing parameters to text file
        print(f"\nOptimization completed. Results saved to {search_output_filename}, {search_txtoutput_filename}.")

try:
    params = best_params
except: #if above fails, ask for input
    params = input(f'\nInsert the parameters for the algorithm {name}:\n(You might find some at analysis/(name of your dataset)/)\n')
    try:
        params = dict(eval(params))
    except:
        print('\nInvalid parameters')
        quit()

optimized_algorithm = alg.set_params(**params)

select_features = input("\nDo you want to select features? 1 or 0.\nYou don't need to select features if your dataset is already a subset of features.\n")
select_features = mm.check_if_int(select_features)

if select_features == 1:
    feature_cv_results, sel_feats, time = mlm.genetic_feature_selection(
        algorithm=optimized_algorithm,
        x_train=data.x_preprocessing,
        y_train=data.y_preprocessing,
        scoring=scoring,
        algorithm_name=name,
        population_size=pop,
        generations=gen)
    feature_output_filename = mm.generate_unique_filename(
        results_path, name, 'feature_selection')
    feature_txtoutput_filename = mm.generate_unique_filename(
        results_path, name, 'selected_features', 'time', suffix='.txt')
    feature_cv_results.to_csv(feature_output_filename)
    with open(feature_txtoutput_filename,'w') as file:
        file.write(f'Selected features:\n{data.x_preprocessing.columns[sel_feats]}\n\nTime to run: {time} seconds.\n\nIn case you need to filter a dataset for columns, use:\ndataset = dataset[list of selected columns]')
    print(f'\nFeature selection completed. Results saved to {feature_output_filename}, {feature_txtoutput_filename}')
    feature_dataset_path = mm.generate_unique_filename(
            datasets_path, name[0:8], 'feat_sel',suffix='')
    print(f'Saving dataset with selected features at {feature_dataset_path}')
    data.x_train = data.x_train[data.x_train.columns[sel_feats]]
    data.x_test = data.x_test[data.x_test.columns[sel_feats]]
    data.x_preprocessing = data.x_preprocessing[data.x_preprocessing.columns[sel_feats]]
    data.save(feature_dataset_path)

score_dfs = []

outlier_detection = input("\nDo you want to detect outliers? 1 or 0.\nYou don't need to detect outliers if your dataset was already filtered.\n")
outlier_detection = mm.check_if_int(outlier_detection)

if outlier_detection == 1:
    print('\nPerforming supervised outlier removal')
    x_train_clean, y_train_clean, cv_results = mlm.supervised_outlier_removal(
        algorithm=optimized_algorithm, x_train=data.x_train, y_train= data.y_train,
        scoring=scoring, algorithm_name=name)
    print('Number of samples before cleaning:', data.x_train.shape[0])
    print('Number of samples after cleaning:', x_train_clean.shape[0])
    print(f'Removed {round(((data.x_train.shape[0]-x_train_clean.shape[0])/data.x_train.shape[0]*100),2)}% of samples')
    outlier_gen_mask = np.logical_not(data.general_data.index.isin(x_train_clean.index))
    outlier_general = data.general_data[outlier_gen_mask]
    outlier_target_mask = np.logical_not(data.x_train.index.isin(x_train_clean.index))
    outlier_target = data.y_train[outlier_target_mask]
    outlier_df = pd.concat([outlier_general, outlier_target],axis=1)
    outlier_dataset_path = mm.generate_unique_filename(
            datasets_path, name[0:8], 'outliers', suffix='')
    print(f'Saving dataset without outliers at {outlier_dataset_path}')
    data.x_train = x_train_clean
    data.y_train = y_train_clean
    data.save(outlier_dataset_path)
    print(f'Outliers are available at {outlier_dataset_path}')
    
    print('\nEvaluating model with cleaned data')
    cv_results_clean = model_selection.cross_validate(
        estimator=optimized_algorithm, X=x_train_clean, y=y_train_clean, cv=10,
        scoring=scoring, n_jobs=-1, return_train_score=True)
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

    score_dfs.extend([score_df_cv_clean, score_df_train_clean])
else:
    print('\nEvaluating model')
    cv_results = model_selection.cross_validate(
        estimator=optimized_algorithm, X=data.x_train, y=data.y_train, cv=10,
        scoring=scoring, n_jobs=-1, return_train_score=True)
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

score_dfs.extend([score_df_cv, score_df_train])
score_df_final = pd.concat(score_dfs, axis=0)
print(score_df_final)
score_output_filename = mm.generate_unique_filename(
    results_path, name, 'scores')
score_df_final.to_csv(score_output_filename)
print(f'\nResults are available at {score_output_filename}')

print('\nFitting model with training data')
model = optimized_algorithm.fit(X=data.x_train, y=data.y_train)
print('Fitting completed')
model_output_filename = mm.generate_unique_filename(
    results_path, name, 'model', suffix='.pkl')
print(f'Saving fitted model to {model_output_filename}')
with open(model_output_filename, 'wb') as file:
    pickle.dump(model, file)
print('Model saved')
test_model = input('\nDo you want to test the model? 1 or 0.\n')
test_model = mm.check_if_int(test_model)
if test_model == 1:
    print('\nEvaluating model with test data.\n\nAttention!\n\nDo not use these results to optimize your model, lest you should leak information from the test data into the model.')
    y_pred_test = model.predict(data.x_test)
    r2_test = metrics.r2_score(data.y_test, y_pred_test)
    rmse_test = metrics.root_mean_squared_error(data.y_test, y_pred_test)
    mae_test = metrics.mean_absolute_error(data.y_test, y_pred_test)
    score_df_test = pd.DataFrame(
        {'r2':r2_test, 'rmse':rmse_test, 'mae':mae_test}, index=['score_test'])
    test_output_filename = mm.generate_unique_filename(
            results_path, name, 'test')
    score_df_test.to_csv(test_output_filename)
    print(f'\nTest results are available at {test_output_filename}')
    
print('\nExiting')
# trained models output removed, as it is not the point of this analysis

# model_output_filename = mm.generate_unique_filename(
#     results_path, algorithm[0], 'Model',suffix='.pkl')
# joblib.dump(model, model_output_filename)
# model_clean_output_filename = mm.generate_unique_filename(
#     results_path, algorithm[0], 'ModelClean',suffix='.pkl')
# joblib.dump(model_clean, model_clean_output_filename)

