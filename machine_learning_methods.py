import sklearn.model_selection as model_selection
import numpy as np
import pandas as pd
import time
import miscelanneous_methods as mm
import os


def supervised_outlier_removal(algorithm, x_train, y_train, scoring, algorithm_name, cv = 10):
    cv_results = model_selection.cross_validate(estimator=algorithm, X=x_train, y=y_train,
                                                cv=cv, scoring=scoring, return_estimator=True,
                                                return_indices=True)
    residues = pd.Series()

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
    print('Number of samples before cleaning:', x_train.shape[0])
    print('Number of samples after cleaning:', x_train_clean.shape[0])
    print(f'\nRemoved {round(((x_train.shape[0]-x_train_clean.shape[0])/x_train.shape[0]*100),2)}% of samples')
    return x_train_clean, y_train_clean, cv_results


def evaluate_and_optimize(algorithm, params, x_train, y_train, scoring, algorithm_name):
    start_time = time.time()
    print(f"\nOptimizing {algorithm_name}")
    print(f"Parameters: {params}")
    grid_search = model_selection.GridSearchCV(
        estimator=algorithm, param_grid=params,
        scoring=scoring, refit='r2', n_jobs=-1)
    print('\nFitting\n')
    grid_search.fit(x_train, y_train)                                   
    search_cv_results = pd.DataFrame(grid_search.cv_results_)
    best_params = grid_search.best_params_
    time_to_execute = time.time()-start_time
    print(f'Results:\n{search_cv_results}\n')
    print(f'Best parameters:\n{best_params}')
    print(f'Best score index:\n{grid_search.best_index_}')
    print(f'Time to execute: {time_to_execute} seconds')
    search_cv_results['time_to_execute'] = time_to_execute
    # search_output_filename = mm.generate_unique_filename(results_path, algorithm_name, 'GridSearch')
    # search_cv_results.to_csv(search_output_filename, index=False)
   
    return search_cv_results, best_params
