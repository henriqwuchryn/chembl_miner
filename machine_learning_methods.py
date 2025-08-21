import sklearn.model_selection as model_selection
from sklearn_genetic.callbacks import DeltaThreshold
from sklearn_genetic import GASearchCV, GAFeatureSelectionCV
from sklearn_genetic.space import Categorical, Integer, Continuous, Space
import numpy as np
import pandas as pd
import time
import miscelanneous_methods as mm
import os


def supervised_outlier_removal(
    algorithm, x_train, y_train, scoring, algorithm_name, cv=10
):
    cv_results = model_selection.cross_validate(
        estimator=algorithm,
        X=x_train,
        y=y_train,
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        return_estimator=True,
        return_indices=True,
        return_train_score=True,
    )
    residues = pd.Series().astype(float)

    for fold in range(cv):
        x_test_fold = x_train.iloc[cv_results["indices"]["test"][fold]]
        y_test_fold = y_train.iloc[cv_results["indices"]["test"][fold]]
        y_pred_fold = cv_results["estimator"][fold].predict(x_test_fold)
        residue_fold = y_test_fold - y_pred_fold
        residues = pd.concat([residues, residue_fold])

    residue_std = np.std(residues)
    limit = 2 * residue_std
    outlier_mask = np.abs(residues) <= limit
    x_train_clean = x_train[outlier_mask]
    y_train_clean = y_train[outlier_mask]
    return x_train_clean, y_train_clean, cv_results


def evaluate_and_optimize(
    algorithm,
    param_grid,
    x_train,
    y_train,
    scoring,
    algorithm_name,
    population_size=30,
    generations=30,
    refit="r2",
):
    start_time = time.time()
    print(f"\nOptimizing {algorithm_name}")
    print(f"Parameters: {describe_params(param_grid)}")
    param_search = GASearchCV(
        estimator=algorithm,
        param_grid=param_grid,
        scoring=scoring,
        population_size=population_size,
        generations=generations,
        refit=refit,
        n_jobs=-1,
        return_train_score=True,
    )
    print("\nFitting")
    callback = DeltaThreshold(threshold=0.001, generations=3)
    param_search.fit(x_train, y_train, callbacks=callback)
    search_cv_results = pd.DataFrame(param_search.cv_results_)
    print(f"\nResults:\n{search_cv_results}\n")
    best_params = param_search.best_params_
    print(f"Best parameters:\n{best_params}")
    print(f"Best score index:\n{param_search.best_index_}")
    time_to_execute = time.time() - start_time
    print(f"Time to execute: {time_to_execute} seconds")
    return search_cv_results, best_params, time_to_execute


def genetic_feature_selection(
    algorithm,
    x_train,
    y_train,
    scoring,
    algorithm_name,
    population_size=30,
    generations=30,
    refit="r2",
):
    start_time = time.time()
    print(f"\nSelecting features for {algorithm_name}")
    feature_selection = GAFeatureSelectionCV(
        estimator=algorithm,
        scoring=scoring,
        population_size=population_size,
        generations=generations,
        refit=refit,
        n_jobs=-1,
        return_train_score=True,
    )
    print("\nFitting")
    callback = DeltaThreshold(threshold=0.001, generations=3)
    feature_selection.fit(x_train, y_train, callback)
    feature_cv_results = pd.DataFrame(feature_selection.cv_results_)
    print(f"\nResults:\n{feature_cv_results}\n")
    selected_features = feature_selection.support_
    print(f"Selected features:\n{x_train.columns[selected_features]}")
    time_to_execute = time.time() - start_time
    print(f"Time to execute: {time_to_execute} seconds")
    return feature_cv_results, selected_features, time_to_execute


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
    param_string = ""
    for param in params:
        value = params[param]
        if type(value) == Categorical:
            line = f"{param}: {value.choices}"
        else:
            line = f"{param}: {value.lower}-{value.upper}"
        param_string = param_string + "\n" + line
    return param_string


def get_results(cv_results, name=""):
    score_dfs = []
    if name != "":
        name = f"_{name}"
    try:
        r2_test_mean = cv_results["test_r2"].mean()
        r2_test_std = cv_results["test_r2"].std()
        r2_train_mean = cv_results["train_r2"].mean()
        r2_train_std = cv_results["train_r2"].std()
        r2_df = pd.DataFrame(
            {
                "test_mean": r2_test_mean,
                "test_std": r2_test_std,
                "train_mean": r2_train_mean,
                "train_std": r2_train_std,
            },
            index=[f"r2{name}"],
        )
        score_dfs.extend(r2_df)
    except KeyError as e:
        print("Not available: r2")

    try:
        rmse_test_mean = cv_results["test_rmse"].mean()
        rmse_test_std = cv_results["test_rmse"].std()
        rmse_train_mean = cv_results["train_rmse"].mean()
        rmse_train_std = cv_results["train_rmse"].std()
        rmse_df = pd.DataFrame(
            {
                "test_mean": rmse_test_mean,
                "test_std": rmse_test_std,
                "train_mean": rmse_train_mean,
                "train_std": rmse_train_std,
            },
            index=[f"rmse{name}"],
        )
        score_dfs.extend(rmse_df)
    except KeyError as e:
        print("Not available: rmse")

    try:
        mae_test_mean = cv_results["test_mae"].mean()
        mae_test_std = cv_results["test_mae"].std()
        mae_train_mean = cv_results["train_mae"].mean()
        mae_train_std = cv_results["train_mae"].std()
        mae_df = pd.DataFrame(
            {
                "test_mean": mae_test_mean,
                "test_std": mae_test_std,
                "train_mean": mae_train_mean,
                "train_std": mae_train_std,
            },
            index=[f"mae{name}"],
        )
        score_dfs.extend(mae_df)
    except KeyError as e:
        print("Not available: mae")

    try:
        quantile_test_mean = cv_results["test_quantile"].mean()
        quantile_test_std = cv_results["test_quantile"].std()
        quantile_train_mean = cv_results["train_quantile"].mean()
        quantile_train_std = cv_results["train_quantile"].std()
        quantile_df = pd.DataFrame(
            {
                "test_mean": quantile_test_mean,
                "test_std": quantile_test_std,
                "train_mean": quantile_train_mean,
                "train_std": quantile_train_std,
            },
            index=[f"quantile{name}"],
        )
        score_dfs.extend(quantile_df)
    except KeyError as e:
        print("Not available: quantile")
    return score_dfs
