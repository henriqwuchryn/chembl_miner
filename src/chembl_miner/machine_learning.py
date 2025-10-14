import numpy as np
import pandas as pd
import joblib
from deap.tools import crossover, mutation
from sklearn.base import BaseEstimator
from sklearn.ensemble import BaggingRegressor, ExtraTreesRegressor, GradientBoostingRegressor, \
    HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import make_scorer, r2_score, root_mean_squared_error, mean_absolute_error, mean_pinball_loss
from sklearn.metrics._scorer import _BaseScorer
from sklearn.model_selection import cross_validate
from sklearn.svm import OneClassSVM
from sklearn_genetic import DeltaThreshold, GASearchCV
from sklearn_genetic.space import Integer, Continuous, Categorical
from xgboost import XGBRegressor

from .datasets import TrainingData, PredictionData
from .utils import print_low, print_high, _check_kwargs


class ModelPipeline:

    def __init__(
            self,
            algorithm_name: str = None,
            algorithm: BaseEstimator = None,
            fit_model: BaseEstimator = None,
            applicability_domain_model: OneClassSVM = None,
            scoring: dict = None,
            param_grid: dict = None,
            params: dict = None,
    ):
        self.algorithm_name = algorithm_name
        self.algorithm = algorithm
        self.fit_model = fit_model
        self.applicability_domain_model = OneClassSVM if applicability_domain_model is None else applicability_domain_model
        self.scoring = {} if scoring is None else scoring
        self.param_grid = {} if param_grid is None else param_grid
        self.params = {} if params is None else params

    @classmethod
    def setup(
            cls,
            algorithm: BaseEstimator | str,
            scoring: dict | str | list[str] = ["r2", "rmse", "mae"],
            random_state: int = 42,
            n_jobs: int = -1,
            **scoring_params,
    ):
        print_low("Setting up MLWrapper object.")
        instance = cls()
        instance._set_algorithm(
            algorithm=algorithm,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        print_high(f"Algorithm set to: {instance.algorithm_name}")
        print_high(f"Random state: {random_state}, n_jobs: {n_jobs}")

        alpha = _check_kwargs(kwargs=scoring_params, arg='alpha', default=0.5, type_to_check=float)

        instance._set_scoring(scoring=scoring, alpha=alpha)
        print_high(f"Scoring metrics set to: {list(instance.scoring.keys())}")
        print_low("✅ MLWrapper setup complete.")

        return instance

    def save_model_pipeline(self, file_path: str) -> None:
        """
        Saves the ModelPipeline object to a file using joblib.

        Args:
            file_path (str): The path to the file where the object will be saved.
                             It's common to use a '.joblib' or '.pkl' extension.
        """
        try:
            print_low(f"Saving ModelPipeline object to {file_path}.")
            joblib.dump(self, file_path)
            print_low("Pipeline saved successfully.")
        except Exception as e:
            print(f"Error saving pipeline: {e}")
            raise e

    @staticmethod
    def load_model_pipeline(file_path: str):
        """
        Loads a ModelPipeline object from a file using joblib.

        Args:
            file_path (str): The path to the file containing the saved object.

        Returns:
            ModelPipeline: The loaded ModelPipeline object.
        """
        try:
            print_low(f"Loading ModelPipeline object from {file_path}...")
            pipeline = joblib.load(file_path)
            if not isinstance(pipeline, ModelPipeline):
                print_low("Warning: Loaded object is not an instance of ModelPipeline.")
            else:
                print_low("Pipeline loaded successfully.")
            return pipeline
        except FileNotFoundError:
            print(f"Error: No file found at {file_path}")
            raise
        except Exception as e:
            print(f"Error loading pipeline: {e}")
            raise e

    # TODO: implementar outros métodos de busca (grid, random)
    def optimize_hyperparameters(
            self,
            dataset: TrainingData,
            cv: int = 3,
            param_grid: dict | None = None,
            refit: str | bool = True,
            population_size: int = 30,
            generations: int = 30,
            n_jobs=-1,
            **kwargs
    ):
        print_low("Starting hyperparameter optimization with GASearchCV (genetic algorithm parameter search).")
        self._check_attributes()
        if dataset.x_train.empty:
            raise ValueError("Dataset empty.")
        if (self.algorithm_name is None) and (param_grid is None):
            raise ValueError(
                "A param_grid was not provided. Provide a param_grid compatible with sklearn_genetic (https://sklearn-genetic-opt.readthedocs.io/en/stable/api/space.html)\nAlternatively, provide algorithm_name, to use one of the param_grids provided by the package.",
            )
        print_high(f"CV folds: {cv}, Population: {population_size}, Generations: {generations}")
        if refit:
            print_high(f"Refit: {refit}. If using multiple metrics, will not provide a final model or best_params . ")
        elif isinstance(refit, str):
            print_high(f"Refitting model based on best '{refit}' score.")
        else:
            print_high(f"Refit: {refit}. Will not provide a final model or best_params. ")

        if param_grid is None:
            available_param_grids: dict = {
                "bagging_reg": {
                    "n_estimators": Integer(lower=10, upper=1000),  # 10
                    "max_samples": Continuous(lower=0.1, upper=1.0),  # 1.0
                    "max_features": Continuous(lower=0.1, upper=1.0),  # 1.0
                    "bootstrap": Categorical(
                        choices=[True, False],
                    ),  # Whether samples are drawn with replacement
                    "bootstrap_features": Categorical(
                        choices=[True, False],
                    ),  # Whether features are drawn with replacement
                },
                "extratrees_reg": {
                    "n_estimators": Integer(lower=100, upper=2000),  # 100
                    "max_depth": Categorical([None]),  # None
                    "min_samples_split": Integer(lower=2, upper=20),  # 2
                    "min_samples_leaf": Integer(lower=1, upper=20),  # 1
                    "max_features": Continuous(
                        lower=0.1,
                        upper=1,
                    ),  # Number of features to consider for splits
                    "bootstrap": Categorical(
                        choices=[True, False],
                    ),  # Whether bootstrap samples are used
                },
                "gradboost_reg": {
                    "n_estimators": Integer(lower=100, upper=2000),  # 100
                    "learning_rate": Continuous(lower=0.001, upper=1),  # 0.1
                    "max_depth": Integer(lower=3, upper=100),  # 3
                    "min_samples_split": Integer(lower=2, upper=20),  # 2
                    "min_samples_leaf": Integer(lower=1, upper=20),  # 1
                    "subsample": Continuous(lower=0.1, upper=1.0),  # 1.0
                    "max_features": Continuous(lower=0.1, upper=1.0),  # 1.0
                },
                "histgradboost_reg": {
                    "loss": Categorical(
                        choices=["squared_error", "absolute_error"],
                    ),  # squared_error
                    "max_iter": Integer(lower=100, upper=2000),  # 100
                    "learning_rate": Continuous(lower=0.001, upper=1),
                    "max_depth": Categorical(choices=[None]),  # None
                    "min_samples_leaf": Integer(lower=10, upper=200),  # 20
                    "max_leaf_nodes": Integer(lower=10, upper=200),  # 61
                    "l2_regularization": Continuous(lower=0.1, upper=2.0),  # 0
                    "max_bins": Integer(lower=100, upper=255),  # 255
                },
                "lgbm_reg": {
                    "n_estimators": Integer(lower=100, upper=2000),  # 100
                    "learning_rate": Continuous(lower=0.001, upper=1),  # 0.1
                    "max_depth": Integer(lower=3, upper=100),  # -1
                    "num_leaves": Integer(lower=2, upper=200),  # 31
                    "min_child_samples": Integer(lower=2, upper=200),  # 20
                    "subsample": Continuous(
                        lower=0.1,
                        upper=1,
                    ),  # Fraction of samples used for fitting
                    "colsample_bytree": Continuous(
                        lower=0.1,
                        upper=1,
                    ),  # Fraction of features used for fitting
                    "reg_alpha": Continuous(lower=0.1, upper=2.0),  # L1 regularization
                    "reg_lambda": Continuous(lower=0.1, upper=2.0),  # L2 regularization
                    "force_row_wise": Categorical(choices=[True]),
                },
                "randomforest_reg": {
                    "n_estimators": Integer(lower=100, upper=2000),  # 100
                    "max_depth": Categorical([None]),  # None
                    "min_samples_split": Integer(lower=2, upper=20),  # 2
                    "min_samples_leaf": Integer(lower=1, upper=20),  # 1
                    "max_features": Continuous(lower=0.1, upper=1),  # 1.0
                    "bootstrap": Categorical(
                        choices=[True, False],
                    ),  # Whether bootstrap samples are used
                },
                "xgboost_reg": {
                    "objective": Categorical(
                        choices=["reg:squarederror", "reg:squaredlogerror","reg:pseudohubererror",
                                 "reg:absoluteerror"]),
                    "n_estimators": Integer(lower=100, upper=2000),
                    "learning_rate": Continuous(lower=0.001, upper=1),
                    "max_depth": Integer(lower=0, upper=100),
                    "min_child_weight": Continuous(
                        lower=0.1,
                        upper=2.0,
                    ),  # Minimum sum of instance weight needed in a child
                    "gamma": Continuous(
                        lower=0.1,
                        upper=2.0,
                    ),  # Minimum loss reduction required to make a split
                    "subsample": Continuous(
                        lower=0.1,
                        upper=1,
                    ),  # Fraction of samples used for fitting
                    "colsample_bytree": Continuous(
                        lower=0.1,
                        upper=1,
                    ),  # Fraction of features used for fitting
                    "reg_alpha": Continuous(lower=0.1, upper=2.0),  # L1 regularization
                    "reg_lambda": Continuous(lower=0.1, upper=2.0),  # L2 regularization
                },
                "xgboost_reg_quantile": {
                    "objective": Categorical(
                        choices=["reg:quantileerror"]),
                    "quantile_alpha": Continuous(lower=0.01, upper=0.99),
                    "n_estimators": Integer(lower=100, upper=2000),
                    "learning_rate": Continuous(lower=0.001, upper=1),
                    "max_depth": Integer(lower=0, upper=100),
                    "min_child_weight": Continuous(
                        lower=0.1,
                        upper=2.0,
                    ),  # Minimum sum of instance weight needed in a child
                    "gamma": Continuous(
                        lower=0.1,
                        upper=2.0,
                    ),  # Minimum loss reduction required to make a split
                    "subsample": Continuous(
                        lower=0.1,
                        upper=1,
                    ),  # Fraction of samples used for fitting
                    "colsample_bytree": Continuous(
                        lower=0.1,
                        upper=1,
                    ),  # Fraction of features used for fitting
                    "reg_alpha": Continuous(lower=0.1, upper=2.0),  # L1 regularization
                    "reg_lambda": Continuous(lower=0.1, upper=2.0),  # L2 regularization
                },
            }
            if self.algorithm_name not in available_param_grids.keys():
                raise ValueError(
                    'provided algorithm_name does not have a param_grid available.\nPlease, provide a param_grid compatible with sklearn_genetic (https://sklearn-genetic-opt.readthedocs.io/en/stable/api/space.html)',
                )
            print_high(f"Using pre-defined parameter grid for '{self.algorithm_name}'.")
            param_grid = available_param_grids[self.algorithm_name]
        else:
            print_high("Using user-provided parameter grid.")
        try:
            crossover_probability = _check_kwargs(kwargs,'crossover_probability',0.2)
            mutation_probability = _check_kwargs(kwargs,'mutation_probability',0.8)
            tournament_size = _check_kwargs(kwargs,'tournament_size',3)
            callback = DeltaThreshold(threshold=0.001, generations=3)
            param_search = GASearchCV(
                estimator=self.algorithm,
                cv=cv,
                param_grid=param_grid,
                scoring=self.scoring,
                population_size=population_size,
                generations=generations,
                crossover_probability=crossover_probability,
                mutation_probability=mutation_probability,
                tournament_size=tournament_size,
                refit=refit,  # type: ignore
                n_jobs=n_jobs,
                return_train_score=True,
            )
            param_search.fit(dataset.x_train, dataset.y_train, callbacks=callback)
            print_low("Hyperparameter optimization complete.")
        except Exception as e:
            print("Something went wrong during optimization.")
            raise e
        try:
            self.params = param_search.best_params_
        except AttributeError as e:
            print("Best parameters were not defined because no refit method (string with scorer name) was provided.")
            print_low("Check resulting param_search to determine best parameters, or rerun with refit method", )
            print(e)
        return param_search

    def evaluate_model(
            self,
            dataset: TrainingData,
            cv: int = 10,
            params: dict | None = None,
            n_jobs=-1,
    ):
        print_low(f"Starting model evaluation with {cv}-fold cross-validation...")
        self._check_attributes()
        if dataset.x_train.empty:
            raise ValueError("Dataset empty")
        if params is not None:
            print_high("Using provided parameters for evaluation.")
            _algorithm = self.algorithm.set_params(**params)
        elif self.params != {}:
            print_high("Using optimized parameters found from hyperparameter optimization.")
            _algorithm = self.algorithm.set_params(**self.params)
        else:
            print_low("No provided parameters")
            _algorithm = self.algorithm

        print_high(f"Model parameters for CV: {_algorithm.get_params()}")
        cv_results = cross_validate(
            estimator=_algorithm,
            X=dataset.x_train,
            y=dataset.y_train,
            cv=cv,
            scoring=self.scoring,
            n_jobs=n_jobs,
            return_train_score=True,
        )
        print_low("Cross-validation complete.")
        # TODO: add a way to visualize cv results with generalization
        return cv_results

    def unpack_cv_results(self, cv_results):
        """
        Unpacks cv_results into a long-format DataFrame.
        """
        results_list = []
        for scorer_name in self.scoring.keys():
            test_key = f'test_{scorer_name}' if f'test_{scorer_name}' in cv_results else 'test_score'
            train_key = f'train_{scorer_name}' if f'train_{scorer_name}' in cv_results else 'train_score'
            results_list.append(
                {
                    'scorer': scorer_name,
                    'dataset_type': 'test',
                    'mean': np.mean(cv_results[test_key]),
                    'sd': np.std(cv_results[test_key]),
                },
            )
            if train_key in cv_results:
                results_list.append(
                    {
                        'scorer': scorer_name,
                        'dataset_type': 'train',
                        'mean': np.mean(cv_results[train_key]),
                        'sd': np.std(cv_results[train_key]),
                    },
                )

        return pd.DataFrame(results_list)

    def fit(
            self,
            dataset: TrainingData,
            params: dict | None = None,
    ):
        print_low("Fitting model on the training dataset.")
        self._check_attributes()
        if dataset.x_train.empty:
            raise ValueError("Dataset empty")
        if params is not None:
            print_high("Using provided parameters for evaluation.")
            _algorithm = self.algorithm.set_params(**params)
        elif self.params != {}:
            print_high("Using optimized parameters found from hyperparameter optimization.")
            _algorithm = self.algorithm.set_params(**self.params)
        else:
            print("No provided parameters")
            _algorithm = self.algorithm

        print_high(f"Final model parameters: {_algorithm.get_params()}")
        fit_model = _algorithm.fit(X=dataset.x_train, y=dataset.y_train)
        self.fit_model = fit_model
        print_low("Model fitting complete.")
        return fit_model

    def fit_applicability_domain(self, dataset: TrainingData, **kwargs):
        """
        Fits a One-Class SVM model to the training data to define the applicability domain.
        Any additional keyword arguments are passed directly to the OneClassSVM constructor.
        """
        print_low("Fitting Applicability Domain model (OneClassSVM).")
        if dataset.x_train.empty:
            raise ValueError("Training data in the dataset is empty.")

        gamma = _check_kwargs(kwargs=kwargs, arg='gamma', default='scale')

        self.applicability_domain_model = OneClassSVM(gamma=gamma)
        self.applicability_domain_model.fit(dataset.x_train)
        print_low("Applicability Domain model fitting complete.")

    def _check_training_data_leakage(
            self,
            training_features: pd.DataFrame,
            deployment_features: pd.DataFrame,
    ) -> pd.Series:
        """
        Checks for exact matches between deployment and training feature sets.
        Returns a boolean Series indicating if a deployment sample is in the training set.
        """
        print_low("Checking for data leakage from training set...")
        # Convert training dataframe to a set of tuples for efficient comparison
        train_set = set([tuple(row) for row in training_features.values])
        deploy_list = [tuple(row) for row in deployment_features.values]

        # Check for membership
        is_in_training = [item in train_set for item in deploy_list]
        leakage_count = sum(is_in_training)
        if leakage_count > 0:
            print_high(f"Warning: Found {leakage_count} exact matches from the training data in the deployment set.")
        else:
            print_high("No data leakage found.")
        is_in_training = pd.Series(is_in_training, index=deployment_features.index)
        return is_in_training

    def deploy(
            self,
            deploy_dataset: PredictionData,
            training_dataset: TrainingData,
    ):
        print_low("Deploying model and making predictions...")
        if self.fit_model is None:
            print('Model not fit. Please use the .fit() method first.')
            return None
        if deploy_dataset.deploy_descriptors.empty:
            print(
                'Deployment dataset provided does not contain descriptors. Please use prepare_dataset() or prepare_deploy_dataset() methods.',
            )
            return None

        deploy_dataset.deploy_data['in_training_set'] = self._check_training_data_leakage(
            training_features=training_dataset.x_train,
            deployment_features=deploy_dataset.deploy_descriptors,
        )

        if self.applicability_domain_model:
            print_high("Assessing applicability domain...")
            in_domain = self.applicability_domain_model.predict(deploy_dataset.deploy_descriptors)
            # Convert to boolean: 1 (inlier) -> True, -1 (outlier) -> False
            deploy_dataset.deploy_data['in_applicability_domain'] = (in_domain == 1)
            outliers_count = (in_domain == -1).sum()
            if outliers_count > 0:
                print_low(f"Warning: {outliers_count} deployment samples are outside the applicability domain.")
            else:
                print_high("All deployment samples are within the applicability domain.")
        else:
            print_low("Applicability domain model not fitted. Skipping this check.")

        print_high(f"Predicting on {deploy_dataset.deploy_descriptors.shape[0]} samples.")
        prediction = self.fit_model.predict(deploy_dataset.deploy_descriptors)
        deploy_dataset.prediction = prediction
        if len(prediction) != deploy_dataset.deploy_data.shape[0]:
            print_low("Prediction shape does not match deploy_data.shape. Check descriptors for missing values.")
        else:
            deploy_dataset.deploy_data.loc[:, f'{self.algorithm_name}_prediction'] = prediction
            print_high(f"Predictions added to deploy_data under column '{self.algorithm_name}_prediction'.")
        print_low("✅ Prediction complete.")
        return None

    def _set_algorithm(
            self,
            algorithm: str | BaseEstimator,
            random_state: int,
            n_jobs: int,
    ) -> None:

        if isinstance(algorithm, str):
            available_algorithms: dict = {
                "bagging_reg": BaggingRegressor(random_state=random_state, n_jobs=n_jobs),
                "extratrees_reg": ExtraTreesRegressor(random_state=random_state, n_jobs=n_jobs),
                "gradboost_reg": GradientBoostingRegressor(random_state=random_state),
                "histgradboost_reg": HistGradientBoostingRegressor(
                    random_state=random_state,
                ),
                "randomforest_reg": RandomForestRegressor(random_state=random_state, n_jobs=n_jobs),
                "xgboost_reg": XGBRegressor(random_state=random_state, n_jobs=n_jobs),
                "xgboost_reg_quantile": XGBRegressor(random_state=random_state, n_jobs=n_jobs),
                }
            if algorithm not in available_algorithms.keys():
                raise ValueError(f'Algorithm {algorithm} not recognized')
            self.algorithm = available_algorithms[algorithm]
            self.algorithm_name = algorithm
        elif isinstance(algorithm, BaseEstimator):
            self.algorithm = algorithm
            self.algorithm_name = algorithm.__class__.__name__
            print_low(
                "Package functionality might not work properly with external algorithms",
            )
        else:
            raise TypeError("Input must be a string or an unfitted scikit-learn estimator.")

    def _set_scoring(
            self,
            scoring: str | list[str] | dict,  # type: ignore
            alpha: float,
    ) -> None:

        if isinstance(scoring, dict):
            self.scoring = scoring
            for scorer in scoring.values():
                if not isinstance(scorer, _BaseScorer):
                    raise TypeError(
                        'Provided scorer is not a scikit-learn scorer, package functionality might not work properly.\nUse scikit-learn make_scorer function to create a scorer from a function.',
                    )


        elif isinstance(scoring, (str, list)):
            available_scorers: dict = {
                "r2": make_scorer(r2_score),
                "rmse": make_scorer(
                    lambda y_true, y_pred: root_mean_squared_error(y_true, y_pred),
                    greater_is_better=False,
                ),
                "mae": make_scorer(mean_absolute_error, greater_is_better=False),
                "quantile": make_scorer(
                    lambda y_true, y_pred: mean_pinball_loss(
                        y_true,
                        y_pred,
                        alpha=alpha,
                    ),
                    greater_is_better=False,
                ),
            }
            scoring_dict: dict = {}
            if isinstance(scoring, str):
                scoring = [scoring]
            for scoring_name in scoring:
                try:
                    scorer = available_scorers[scoring_name]
                    scoring_dict[scoring_name] = scorer
                except KeyError as e:
                    print_low(f"{scoring_name} is not available as a scoring metric")
                    print_low(e)
            if scoring_dict == {}:
                raise ValueError("No valid scoring metrics found from the provided list.")
            self.scoring = scoring_dict
        else:
            raise TypeError("Input must be a dictionary, string, or list of strings.")
        # TODO: OLHAR ISSO - TESTAR SE CHEGA NO ELSE

    def _check_attributes(
            self,
            # check_params: bool = True,
    ):
        if self.algorithm is None:
            raise ValueError("Algorithm not set. Define algorithm using setup()")
        if self.scoring == {}:
            raise ValueError("Scorers not set. Define scoring using setup()")
        # if check_params:
        #     if self.params == {}:
        #         raise ValueError("define params using setup()")
