import sklearn.model_selection as model_selection
from sklearn_genetic.callbacks import DeltaThreshold
from sklearn_genetic import GASearchCV
from sklearn_genetic.space import Categorical, Integer, Continuous, Space
import numpy as np
import pandas as pd
import time
import miscelanneous_methods as mm
import os


def supervised_outlier_removal(algorithm, x_train, y_train, scoring, algorithm_name, cv = 10):
    cv_results = model_selection.cross_validate(
            estimator=algorithm,
            X=x_train,
            Y=y_train,
            cv=cv,
            scoring=scoring,
            return_estimator=True,
            return_indices=True,
            return_train_score=True)
    residues = pd.Series().astype(float)

    for fold in range(cv):
        x_test_fold = x_train.iloc[cv_results['indices']['test'][fold]]
        y_test_fold = y_train.iloc[cv_results['indices']['test'][fold]]
        y_pred_fold = cv_results['estimator'][fold].predict(x_test_fold)
        residue_fold = y_test_fold - y_pred_fold
        residues = pd.concat([residues, residue_fold])

    residue_std = np.std(residues)
    limit = 2 * residue_std
    outlier_mask = np.abs(residues) <= limit
    x_train_clean = x_train[outlier_mask]
    y_train_clean = y_train[outlier_mask]
    return x_train_clean, y_train_clean, cv_results


def evaluate_and_optimize(algorithm, params, x_train, y_train, scoring, algorithm_name, population_size=30, generations=30):
    start_time = time.time()
    print(f"\nOptimizing {algorithm_name}")
    print(f"Parameters: {describe_params(params)}")
    param_search = GASearchCV(
        estimator=algorithm,
        param_grid=params,
        scoring=scoring,
        population_size=population_size,
        generations=generations,
        refit='r2',
        n_jobs=-1,
        return_train_score=True)
    print('\nFitting\n')
    callback = DeltaThreshold(threshold=0.001,generations=2)
    param_search.fit(x_train, y_train, callbacks=callback)
    search_cv_results = pd.DataFrame(param_search.cv_results_)
    print(f'Results:\n{search_cv_results}\n')
    best_params = param_search.best_params_
    print(f'Best parameters:\n{best_params}')
    print(f'Best score index:\n{param_search.best_index_}')
    time_to_execute = time.time()-start_time
    print(f'Time to execute: {time_to_execute} seconds')
    search_cv_results['time_to_execute'] = time_to_execute
    return search_cv_results, best_params, time_to_execute


def get_model_scores(y_pred, y_test):
    r2 = metrics.r2_score(y_test, y_pred)
    rmse = metrics.root_mean_squared_error(y_test, y_pred)
    mae = metrics.mean_absolute_error(y_test, y_pred)
    return r2, mse, mae


def scale_features(features, scaler):
    features_scaled = scaler.fit_transform(features)
    features_scaled = pd.DataFrame(features_scaled, index=features.index)
    return features_scaled


def describe_params(params):
    param_string = ''
    for param in params:
        value = params[param]
        if type(value) == Categorical:
            line = f'{param}: {value.choices}'
        else:
            line = f'{param}: {value.lower}-{value.upper}'
        param_string = param_string + '\n' + line
    return param_string

