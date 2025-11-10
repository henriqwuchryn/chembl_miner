import dill
import numpy as np
import pandas as pd
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
    """
    A wrapper class for the machine learning modeling pipeline.

    This class handles algorithm setup, hyperparameter optimization
    (via GASearchCV), model evaluation (via cross-validation), final
    model fitting, and deployment/prediction.
    """

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
        """
        Initializes the ModelPipeline object.

        It's recommended to use the `setup` class method to create
        and configure an instance.

        Args:
            algorithm_name (str, optional): Name of the algorithm.
                Defaults to None.
            algorithm (BaseEstimator, optional): The unfitted sklearn
                estimator. Defaults to None.
            fit_model (BaseEstimator, optional): The fitted model.
                Defaults to None.
            applicability_domain_model (OneClassSVM, optional): The fitted
                applicability domain model. Defaults to None.
            scoring (dict, optional): Dictionary of sklearn scorers.
                Defaults to None.
            param_grid (dict, optional): Parameter grid for optimization.
                Defaults to None.
            params (dict, optional): Best parameters found.
                Defaults to None.
        
        Example:
            ```python
            from sklearn.ensemble import RandomForestRegressor
            
            # This is not the recommended way; use ModelPipeline.setup() instead.
            pipeline = ModelPipeline(
                algorithm_name="randomforest_reg",
                algorithm=RandomForestRegressor()
            )
            ```
        """
        self.algorithm_name = algorithm_name
        self.algorithm = algorithm
        self.fit_model = fit_model
        self.applicability_domain_model = applicability_domain_model
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
        """
        Sets up and creates a new ModelPipeline instance.

        This is the recommended way to initialize the pipeline.

        Args:
            algorithm (BaseEstimator | str): An unfitted scikit-learn
                estimator or a string key for a predefined algorithm
                (e.g., "xgboost_reg", "randomforest_reg").
            scoring (dict | str | list[str], optional): Scoring metric(s) to
                use. Can be a list/str of predefined keys (e.g., "r2", "mae")
                or a dictionary of custom scorers. Defaults to
                ["r2", "rmse", "mae"].
            random_state (int, optional): Random state for reproducibility.
                Defaults to 42.
            n_jobs (int, optional): Number of parallel jobs. Defaults to -1.
            **scoring_params: Additional parameters for scoring functions
                (e.g., `alpha=0.5` for "quantile" scoring).

        Returns:
            ModelPipeline: A configured instance of the ModelPipeline.
        
        Example:
            ```python
            # Set up a pipeline with XGBoost and default scoring
            ml_pipeline = ModelPipeline.setup(
                algorithm="xgboost_reg",
                random_state=42
            )
            
            # Set up with RandomForest and custom scoring
            ml_pipeline_rf = ModelPipeline.setup(
                algorithm="randomforest_reg",
                scoring=["r2", "mae"]
            )
            
            # Set up for quantile regression
            ml_pipeline_quantile = ModelPipeline.setup(
                algorithm="lgbm_reg",
                scoring=["quantile", "mae"],
                alpha=0.25 # Passed as scoring_param
            )
            ```
        """
        
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


    def to_path(self, file_path: str) -> None:
        """
        Saves the entire ModelPipeline object to a file using dill.

        Args:
            file_path (str): The path to the output file (e.g., 'model.pkl').
        
        Example:
            ```python
            # Assuming 'ml_pipeline' is a fitted ModelPipeline object
            ml_pipeline.to_path("my_model_pipeline.pkl")
            ```
        """
        try:
            print_low(f"Saving ModelPipeline object to {file_path}.")
            with open(file_path, 'wb') as file:
                file.write(dill.dumps(self))
            print_low("Pipeline saved successfully.")
        except Exception as e:
            print(f"Error saving pipeline: {e}")
            raise e


    @staticmethod
    def from_path(file_path: str):
        """
        Loads a ModelPipeline object from a dill file.

        Args:
            file_path (str): The path to the file (e.g., 'model.pkl').

        Returns:
            ModelPipeline: The loaded ModelPipeline object.
        
        Example:
            ```python
            try:
                loaded_pipeline = ModelPipeline.from_path("my_model_pipeline.pkl")
                print(loaded_pipeline.algorithm_name)
            except FileNotFoundError:
                print("Model file not found.")
            ```
        """
        try:
            print_low(f"Loading ModelPipeline object from {file_path}...")
            with open(file_path, 'rb') as file:
                pipeline = dill.loads(file.read())
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
        **kwargs,
        ):
        """
        Optimizes model hyperparameters using a genetic algorithm (GASearchCV).

        Args:
            dataset (TrainingData): The dataset to use for optimization.
            cv (int, optional): Number of cross-validation folds. Defaults to 3.
            param_grid (dict | None, optional): A `sklearn-genetic-opt`
                compatible parameter grid. If None, a predefined grid for the
                selected `algorithm_name` is used. Defaults to None.
            refit (str | bool, optional): The scorer name to use for refitting
                the best model, or True to use the first scorer, or False to
                not refit. Defaults to True.
            population_size (int, optional): GA population size. Defaults to 30.
            generations (int, optional): Number of GA generations. Defaults to 30.
            n_jobs (int, optional): Number of parallel jobs. Defaults to -1.
            **kwargs: Additional arguments for GASearchCV (e.g.,
                `crossover_probability`).

        Returns:
            GASearchCV: The fitted GASearchCV object.
        
        Example:
            ```python
            # Assuming 'ml_pipeline' is a setup ModelPipeline
            # and 'my_dataset' is a populated TrainingData object
            
            # Use 'mae' as the metric to refit the best model
            param_search_results = ml_pipeline.optimize_hyperparameters(
                dataset=my_dataset,
                cv=5,
                refit="mae",
                population_size=10, # Smaller for quick example
                generations=5      # Smaller for quick example
            )
            
            print(f"Best params found: {ml_pipeline.params}")
            ```
        """

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
                "bagging_reg"         : {
                    "n_estimators"      : Integer(lower=10, upper=1000),  # 10
                    "max_samples"       : Continuous(lower=0.1, upper=1.0),  # 1.0
                    "max_features"      : Continuous(lower=0.1, upper=1.0),  # 1.0
                    "bootstrap"         : Categorical(
                        choices=[True, False],
                        ),  # Whether samples are drawn with replacement
                    "bootstrap_features": Categorical(
                        choices=[True, False],
                        ),  # Whether features are drawn with replacement
                    },
                "extratrees_reg"      : {
                    "n_estimators"     : Integer(lower=100, upper=2000),  # 100
                    "max_depth"        : Categorical([None]),  # None
                    "min_samples_split": Integer(lower=2, upper=20),  # 2
                    "min_samples_leaf" : Integer(lower=1, upper=20),  # 1
                    "max_features"     : Continuous(
                        lower=0.1,
                        upper=1,
                        ),  # Number of features to consider for splits
                    "bootstrap"        : Categorical(
                        choices=[True, False],
                        ),  # Whether bootstrap samples are used
                    },
                "gradboost_reg"       : {
                    "n_estimators"     : Integer(lower=100, upper=2000),  # 100
                    "learning_rate"    : Continuous(lower=0.001, upper=1),  # 0.1
                    "max_depth"        : Integer(lower=3, upper=100),  # 3
                    "min_samples_split": Integer(lower=2, upper=20),  # 2
                    "min_samples_leaf" : Integer(lower=1, upper=20),  # 1
                    "subsample"        : Continuous(lower=0.1, upper=1.0),  # 1.0
                    "max_features"     : Continuous(lower=0.1, upper=1.0),  # 1.0
                    },
                "histgradboost_reg"   : {
                    "loss"             : Categorical(
                        choices=["squared_error", "absolute_error"],
                        ),  # squared_error
                    "max_iter"         : Integer(lower=100, upper=2000),  # 100
                    "learning_rate"    : Continuous(lower=0.001, upper=1),
                    "max_depth"        : Categorical(choices=[None]),  # None
                    "min_samples_leaf" : Integer(lower=10, upper=200),  # 20
                    "max_leaf_nodes"   : Integer(lower=10, upper=200),  # 61
                    "l2_regularization": Continuous(lower=0.1, upper=2.0),  # 0
                    "max_bins"         : Integer(lower=100, upper=255),  # 255
                    },
                "lgbm_reg"            : {
                    "n_estimators"     : Integer(lower=100, upper=2000),  # 100
                    "learning_rate"    : Continuous(lower=0.001, upper=1),  # 0.1
                    "max_depth"        : Integer(lower=3, upper=100),  # -1
                    "num_leaves"       : Integer(lower=2, upper=200),  # 31
                    "min_child_samples": Integer(lower=2, upper=200),  # 20
                    "subsample"        : Continuous(
                        lower=0.1,
                        upper=1,
                        ),  # Fraction of samples used for fitting
                    "colsample_bytree" : Continuous(
                        lower=0.1,
                        upper=1,
                        ),  # Fraction of features used for fitting
                    "reg_alpha"        : Continuous(lower=0.1, upper=2.0),  # L1 regularization
                    "reg_lambda"       : Continuous(lower=0.1, upper=2.0),  # L2 regularization
                    "force_row_wise"   : Categorical(choices=[True]),
                    },
                "randomforest_reg"    : {
                    "n_estimators"     : Integer(lower=100, upper=2000),  # 100
                    "max_depth"        : Categorical([None]),  # None
                    "min_samples_split": Integer(lower=2, upper=20),  # 2
                    "min_samples_leaf" : Integer(lower=1, upper=20),  # 1
                    "max_features"     : Continuous(lower=0.1, upper=1),  # 1.0
                    "bootstrap"        : Categorical(
                        choices=[True, False],
                        ),  # Whether bootstrap samples are used
                    },
                "xgboost_reg"         : {
                    "objective"       : Categorical(
                        choices=[
                            "reg:squarederror", "reg:squaredlogerror", "reg:pseudohubererror",
                            "reg:absoluteerror",
                            ],
                        ),
                    "n_estimators"    : Integer(lower=100, upper=2000),
                    "learning_rate"   : Continuous(lower=0.001, upper=1),
                    "max_depth"       : Integer(lower=0, upper=100),
                    "min_child_weight": Continuous(
                        lower=0.1,
                        upper=2.0,
                        ),  # Minimum sum of instance weight needed in a child
                    "gamma"           : Continuous(
                        lower=0.1,
                        upper=2.0,
                        ),  # Minimum loss reduction required to make a split
                    "subsample"       : Continuous(
                        lower=0.1,
                        upper=1,
                        ),  # Fraction of samples used for fitting
                    "colsample_bytree": Continuous(
                        lower=0.1,
                        upper=1,
                        ),  # Fraction of features used for fitting
                    "reg_alpha"       : Continuous(lower=0.1, upper=2.0),  # L1 regularization
                    "reg_lambda"      : Continuous(lower=0.1, upper=2.0),  # L2 regularization
                    },
                "xgboost_reg_quantile": {
                    "objective"       : Categorical(
                        choices=["reg:quantileerror"],
                        ),
                    "quantile_alpha"  : Continuous(lower=0.01, upper=0.99),
                    "n_estimators"    : Integer(lower=100, upper=2000),
                    "learning_rate"   : Continuous(lower=0.001, upper=1),
                    "max_depth"       : Integer(lower=0, upper=100),
                    "min_child_weight": Continuous(
                        lower=0.1,
                        upper=2.0,
                        ),  # Minimum sum of instance weight needed in a child
                    "gamma"           : Continuous(
                        lower=0.1,
                        upper=2.0,
                        ),  # Minimum loss reduction required to make a split
                    "subsample"       : Continuous(
                        lower=0.1,
                        upper=1,
                        ),  # Fraction of samples used for fitting
                    "colsample_bytree": Continuous(
                        lower=0.1,
                        upper=1,
                        ),  # Fraction of features used for fitting
                    "reg_alpha"       : Continuous(lower=0.1, upper=2.0),  # L1 regularization
                    "reg_lambda"      : Continuous(lower=0.1, upper=2.0),  # L2 regularization
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
            crossover_probability = _check_kwargs(kwargs, 'crossover_probability', 0.2)
            mutation_probability = _check_kwargs(kwargs, 'mutation_probability', 0.8)
            tournament_size = _check_kwargs(kwargs, 'tournament_size', 3)
            elitism = _check_kwargs(kwargs, 'elitism', True)
            keep_top_k = _check_kwargs(kwargs, 'keep_top_k', 1)
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
                elitism=elitism,
                keep_top_k=keep_top_k,
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
        """
        Evaluates the model using N-fold cross-validation.

        Uses the parameters stored in `self.params` (from optimization)
        unless a new `params` dictionary is provided.

        Args:
            dataset (TrainingData): The dataset to use for CV.
            cv (int, optional): Number of cross-validation folds. Defaults to 10.
            params (dict | None, optional): A parameter dictionary to use.
                If None, uses `self.params`. Defaults to None.
            n_jobs (int, optional): Number of parallel jobs. Defaults to -1.

        Returns:
            dict: The dictionary of results from `sklearn.model_selection.cross_validate`.
        
        Example:
            ```python
            # Assuming 'ml_pipeline' has optimized parameters in 'ml_pipeline.params'
            # and 'my_dataset' is a populated TrainingData object
            
            cv_results = ml_pipeline.evaluate_model(
                dataset=my_dataset,
                cv=10
            )
            
            print(cv_results['test_r2'])
            # [0.5, 0.45, 0.52, ...]
            
            # Or evaluate with a specific set of parameters
            custom_params = {'n_estimators': 50, 'max_depth': 5}
            cv_results_custom = ml_pipeline.evaluate_model(
                dataset=my_dataset,
                cv=5,
                params=custom_params
            )
            ```
        """

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
        Converts the results dictionary from `cross_validate` into a
        long-format DataFrame.

        Args:
            cv_results (dict): The results dictionary from `evaluate_model`.

        Returns:
            pd.DataFrame: A DataFrame with columns ['scorer', 'dataset_type',
                'mean', 'sd'].
        
        Example:
            ```python
            # Assuming 'cv_results' is from evaluate_model
            # cv_results = {
            #     'fit_time': [...],
            #     'test_r2': [0.5, 0.6],
            #     'train_r2': [0.9, 0.91],
            #     'test_mae': [-0.5, -0.4],
            #     'train_mae': [-0.1, -0.11]
            # }
            # ml_pipeline.scoring = {'r2': ..., 'mae': ...} # Must be set
            
            results_df = ml_pipeline.unpack_cv_results(cv_results)
            
            # print(results_df)
            #   scorer dataset_type  mean   sd
            # 0     r2         test  0.55  0.05
            # 1     r2        train  0.905 0.005
            # 2    mae         test -0.45  0.05
            # 3    mae        train -0.105 0.005
            ```
        """

        results_list = []
        for scorer_name in self.scoring.keys():
            test_key = f'test_{scorer_name}' if f'test_{scorer_name}' in cv_results else 'test_score'
            train_key = f'train_{scorer_name}' if f'train_{scorer_name}' in cv_results else 'train_score'
            results_list.append(
                {
                    'scorer'      : scorer_name,
                    'dataset_type': 'test',
                    'mean'        : np.mean(cv_results[test_key]),
                    'sd'          : np.std(cv_results[test_key]),
                    },
                )
            if train_key in cv_results:
                results_list.append(
                    {
                        'scorer'      : scorer_name,
                        'dataset_type': 'train',
                        'mean'        : np.mean(cv_results[train_key]),
                        'sd'          : np.std(cv_results[train_key]),
                        },
                    )

        return pd.DataFrame(results_list)


    def fit(
        self,
        dataset: TrainingData,
        params: dict | None = None,
        ):
        """
        Fits the final model on the entire training dataset.

        Uses `self.params` unless a `params` dictionary is provided.
        The fitted model is stored in `self.fit_model`.

        Args:
            dataset (TrainingData): The dataset to train on.
            params (dict | None, optional): A parameter dictionary to use.
                If None, uses `self.params`. Defaults to None.

        Returns:
            BaseEstimator: The fitted model.
        
        Example:
            ```python
            # Assuming 'ml_pipeline' has optimized parameters in 'ml_pipeline.params'
            # and 'my_dataset' is a populated TrainingData object
            
            fitted_model = ml_pipeline.fit(dataset=my_dataset)
            
            print(ml_pipeline.fit_model)
            # XGBRegressor(...)
            
            # Or fit with custom parameters
            custom_params = {'n_estimators': 100}
            fitted_model_custom = ml_pipeline.fit(
                dataset=my_dataset,
                params=custom_params
            )
            ```
        """

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
        Fits a One-Class SVM model to define the applicability domain (AD).

        The fitted AD model is stored in `self.applicability_domain_model`.

        Args:
            dataset (TrainingData): The dataset to use for fitting the AD.
                The `x_train` features are used.
            **kwargs: Additional keyword arguments passed to the
                `OneClassSVM` constructor (e.g., `gamma`, `nu`).
        
        Example:
            ```python
            # Assuming 'ml_pipeline' is a setup ModelPipeline
            # and 'my_dataset' is a populated TrainingData object
            
            # Fit AD model with a specific kernel and nu
            ml_pipeline.fit_applicability_domain(
                dataset=my_dataset,
                nu=0.1
            )
            
            print(ml_pipeline.applicability_domain_model)
            # OneClassSVM(nu=0.1)
            ```
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
        Internal helper to check for exact feature matches between train and deploy sets.
        This function is called by deploy() function.

        Args:
            training_features (pd.DataFrame): The training features.
            deployment_features (pd.DataFrame): The deployment features.

        Returns:
            pd.Series: A boolean Series indicating if a deployment sample
                was found in the training set.
        
        Example:
            ```python
            # This is an internal method, typically called by deploy()
            
            train_feats = pd.DataFrame({'f1': [1, 2], 'f2': [3, 4]})
            deploy_feats = pd.DataFrame(
                {'f1': [5, 1], 'f2': [6, 3]}, 
                index=['new_mol', 'leaked_mol']
            )
            
            pipeline = ModelPipeline()
            leakage_series = pipeline._check_training_data_leakage(
               train_feats, deploy_feats
            )
            print(leakage_series)
            # new_mol       False
            # leaked_mol     True
            # Name: is_in_training, dtype: bool
            ```
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
        is_in_training = pd.Series(is_in_training, index=deployment_features.index,name='is_in_training')
        return is_in_training


    def deploy(
        self,
        deploy_dataset: PredictionData,
        training_dataset: TrainingData,
        tag: str = ''
        ):
        """
        Deploys the fitted model to make predictions on new data.

        This method:
        1.  Predicts values using `self.fit_model`.
        2.  Checks for data leakage (if deploy samples were in the train set).
        3.  Checks the applicability domain (if `fit_applicability_domain`
            was run).
        4.  Stores the results in `deploy_dataset.prediction` dictionary.

        Args:
            deploy_dataset (PredictionData): The dataset containing new
                molecules to predict.
            training_dataset (TrainingData): The original training dataset
                (used for leakage check).
            tag (str, optional): An optional tag to append to the
                prediction key. Defaults to ''.

        Returns:
            None: The `deploy_dataset` is modified in-place.
        
        Example:
            ```python
            # Assuming 'ml_pipeline' is fitted (has 'fit_model')
            # 'my_dataset' is the TrainingData used for fitting
            # 'my_pred_data' is a populated PredictionData object
            
            # Fit AD model first (optional, but recommended)
            ml_pipeline.fit_applicability_domain(dataset=my_dataset, nu=0.1)
            
            # Deploy the model
            ml_pipeline.deploy(
                deploy_dataset=my_pred_data,
                training_dataset=my_dataset,
                tag="CHEMBL365"
            )
            
            # Results are stored in the PredictionData object
            print(my_pred_data.prediction.keys())
            # dict_keys(['xgboost_reg_CHEMBL365_1'])
            
            print(my_pred_data.prediction['xgboost_reg_CHEMBL365_1'])
            #    predicted_values  in_applicability_domain  is_in_training
            # 0          8.123...                     True           False
            # 1          7.456...                    False           False
            ```
        """

        if tag != '':
            tag = f'_{tag}'
        print_low("Deploying model and making predictions...")
        if self.fit_model is None:
            print('Model not fit. Please use the .fit() method first.')
            return None
        if deploy_dataset.deploy_descriptors.empty:
            print(
                'Deployment dataset provided does not contain descriptors. Please use prepare_dataset() or prepare_deploy_dataset() methods.',
                )
            return None
        count = 1
        while True:
            prediction_id = f'{self.algorithm_name}{tag}_{count}'
            if prediction_id in deploy_dataset.prediction.keys():
                count += 1
            else:
                break

        print_high(f"Predicting on {deploy_dataset.deploy_descriptors.shape[0]} samples.")
        predicted_values = pd.Series(
            data=self.fit_model.predict(deploy_dataset.deploy_descriptors),
            index=deploy_dataset.deploy_descriptors.index,
            name='predicted_values')

        training_data_leakage = self._check_training_data_leakage(
            training_features=training_dataset.x_train,
            deployment_features=deploy_dataset.deploy_descriptors,
            )

        if self.applicability_domain_model:
            print_high("Assessing applicability domain...")
            in_domain = pd.Series(
                data=self.applicability_domain_model.predict(deploy_dataset.deploy_descriptors),
                index=deploy_dataset.deploy_descriptors.index,
                name='in_applicability_domain')
            # Convert to boolean: 1 (inlier) -> True, -1 (outlier) -> False
            in_applicability_domain = (in_domain == 1)
            outliers_count = (in_domain == -1).sum()
            if outliers_count > 0:
                print_low(f"{outliers_count} deployment samples are outside the applicability domain.")
            else:
                print_high("All deployment samples are within the applicability domain.")
            prediction = pd.concat(
                    [predicted_values, in_applicability_domain, training_data_leakage],
                    axis=1,
                    )

        else:
            print_low("Applicability domain model not fitted. Skipping this check.")
            prediction = pd.concat(
                        [predicted_values, training_data_leakage],
                        axis=1,
                        )

        deploy_dataset.prediction[prediction_id] = prediction
        if len(prediction) != deploy_dataset.deploy_data.shape[0]:
            print_low("Prediction shape does not match deploy_data.shape. Check descriptors for missing values.")
        else:
            print_high(f"Predictions added to prediction under key '{prediction_id}'.")
        print_low("✅ Prediction complete.")
        return None


    def _set_algorithm(
        self,
        algorithm: str | BaseEstimator,
        random_state: int,
        n_jobs: int,
        ) -> None:
        """
        Internal helper to set the algorithm estimator from a string or object.
        This function is called by setup() function.

        Args:
            algorithm (str | BaseEstimator): The algorithm to set.
            random_state (int): The random state.
            n_jobs (int): The number of parallel jobs.
        
        Example:
            ```python
            # This is an internal method, typically called by setup()
            pipeline = ModelPipeline()
            pipeline._set_algorithm("randomforest_reg", 42, -1)
            print(pipeline.algorithm)
            # RandomForestRegressor(n_jobs=-1, random_state=42)
            ```
        """

        if isinstance(algorithm, str):
            available_algorithms: dict = {
                "bagging_reg"         : BaggingRegressor(random_state=random_state, n_jobs=n_jobs),
                "extratrees_reg"      : ExtraTreesRegressor(random_state=random_state, n_jobs=n_jobs),
                "gradboost_reg"       : GradientBoostingRegressor(random_state=random_state),
                "histgradboost_reg"   : HistGradientBoostingRegressor(
                    random_state=random_state,
                    ),
                "randomforest_reg"    : RandomForestRegressor(random_state=random_state, n_jobs=n_jobs),
                "xgboost_reg"         : XGBRegressor(random_state=random_state, n_jobs=n_jobs),
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
        """
        Internal helper to set the scoring dictionary.
        This function is called by setup() function.

        Args:
            scoring (str | list[str] | dict): The scoring definition.
            alpha (float): The alpha value for quantile/pinball loss.
        
        Example:
            ```python
            # This is an internal method, typically called by setup()
            pipeline = ModelPipeline()
            
            # Example 1: Using a list of strings
            pipeline._set_scoring(["r2", "mae"], alpha=0.5)
            print(pipeline.scoring.keys())
            # dict_keys(['r2', 'mae'])
            
            # Example 2: Using a quantile
            pipeline._set_scoring("quantile", alpha=0.75)
            print(pipeline.scoring.keys())
            # dict_keys(['quantile'])
            ```
        """

        if isinstance(scoring, dict):
            self.scoring = scoring
            for scorer in scoring.values():
                if not isinstance(scorer, _BaseScorer):
                    raise TypeError(
                        'Provided scorer is not a scikit-learn scorer, package functionality might not work properly.\nUse scikit-learn make_scorer function to create a scorer from a function.',
                        )


        elif isinstance(scoring, (str, list)):
            available_scorers: dict = {
                "r2"      : make_scorer(r2_score),
                "rmse"    : make_scorer(
                    lambda y_true, y_pred: root_mean_squared_error(y_true, y_pred),
                    greater_is_better=False,
                    ),
                "mae"     : make_scorer(mean_absolute_error, greater_is_better=False),
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
        """
        Internal helper to check if the pipeline is set up.

        Example:
            ```python
            # This is an internal method, called by other methods.
            pipeline = ModelPipeline()
            try:
                pipeline._check_attributes()
            except ValueError as e:
                print(e)
            # Algorithm not set. Define algorithm using setup()
            ```
        """
        
        if self.algorithm is None:
            raise ValueError("Algorithm not set. Define algorithm using setup()")
        if self.scoring == {}:
            raise ValueError("Scorers not set. Define scoring using setup()")
        # if check_params:
        #     if self.params == {}:
        #         raise ValueError("define params using setup()")
