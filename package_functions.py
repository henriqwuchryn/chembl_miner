import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from chembl_webresource_client.new_client import new_client
import machine_learning_methods as mlm
import molecules_manipulation_methods as mmm
import miscelanneous_methods as mm


class DatasetWrapper:

    def __init__(
        self, general_data=None, x_train=None, x_test=None, y_train=None, y_test=None
    ):
        """
        Initializes the DatasetWrapper with optional dataframes.
        """
        self.general_data = general_data if general_data is not None else pd.DataFrame()
        self.x_train = x_train if x_train is not None else pd.DataFrame()
        self.x_test = x_test if x_test is not None else pd.DataFrame()
        self.y_train = y_train if y_train is not None else pd.DataFrame()
        self.y_test = y_test if y_test is not None else pd.DataFrame()
        self.x_preprocessing = pd.DataFrame()
        self.y_preprocessing = pd.DataFrame()

    def subset_general_data(self, subset="train") -> pd.DataFrame:
        """
        Retrieves the corresponding rows from general_data for the train or test dataset.

        Args:
            subset (str): 'train' or 'test', indicating which subset to trace. Default value is 'train'.

        Returns:
            pd.DataFrame: Rows from general_data corresponding to the specified subset.
        """
        if subset == "train":
            return self.general_data.loc[self.x_train.index]
        elif subset == "test":
            return self.general_data.loc[self.x_test.index]
        else:
            raise ValueError("Subset must be 'train' or 'test'.")

    def save(self, file_path) -> None:
        """
        Saves the entire dataset to CSV files.
        """

        if not os.path.exists(file_path):
            os.makedirs(file_path)

        self.general_data.to_csv(f"{file_path}/gd.csv", index_label="index")
        self.x_train.to_csv(f"{file_path}/ftr.csv", index_label="index")
        self.x_test.to_csv(f"{file_path}/fte.csv", index_label="index")
        self.y_train.to_csv(f"{file_path}/ttr.csv", index_label="index")
        self.y_test.to_csv(f"{file_path}/tte.csv", index_label="index")
        self.x_preprocessing.to_csv(f"{file_path}/fpr.csv", index_label="index")
        self.y_preprocessing.to_csv(f"{file_path}/tpr.csv", index_label="index")
        print(f"Dataset saved to {file_path}")

    def load(self, file_path) -> None:
        """
        Loads the dataset from CSV files.
        """
        try:
            self.general_data = pd.read_csv(f"{file_path}/gd.csv", index_col="index")
            self.x_train = pd.read_csv(f"{file_path}/ftr.csv", index_col="index")
            self.x_test = pd.read_csv(f"{file_path}/fte.csv", index_col="index")
            self.y_train = pd.read_csv(f"{file_path}/ttr.csv", index_col="index")[
                "neg_log_value"
            ]
            self.y_test = pd.read_csv(f"{file_path}/tte.csv", index_col="index")[
                "neg_log_value"
            ]
            self.x_preprocessing = pd.read_csv(
                f"{file_path}/fpr.csv", index_col="index"
            )
            self.y_preprocessing = pd.read_csv(
                f"{file_path}/tpr.csv", index_col="index"
            )["neg_log_value"]
            self.file_path = file_path
            print(f"Dataset loaded from {file_path}")
        except Exception as e:
            print(e)
            print("Dataset loading failed")

    def load_unsplit_dataframe(
        self,
        full_df,
        target_column,
        nonfeature_columns,
        holdout_size,
        random_state,
        preprocessing_size,
    ) -> None:
        """
        Loads an unsplit dataframe containing all columns and splits it into
        general_data, x_train/test, and y_train/test.

        Args:
            dataframe (pandas.DataFrame): Unsplit dataframe containing the dataset.
            general_columns (list): List of columns representing general data.
            target_column (str): Column name representing the target variable.
            holdout_size (float): Proportion of the dataset to include in the test split. Standard value = 0.2.
            random_state (int): Random state for reproducibility. Standard value = 42.
        """
        self.general_data = full_df[nonfeature_columns]
        features = full_df.drop(columns=nonfeature_columns)
        target = full_df[target_column]
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            features, target, holdout_size=holdout_size, random_state=random_state
        )
        self.file_path = file_path
        print(f"Unsplit dataset loaded and split into train/test sets")
        # Create preprocessing dataset
        self.create_preprocessing_dataset(
            random_state=random_state, preprocessing_size=preprocessing_size
        )

    def create_preprocessing_dataset(
        self, random_state, preprocessing_size=7500
    ) -> None:
        """
        Creates a preprocessing dataset by sampling from the train dataset.
        If the train dataset is larger max_samples, it samples this amount of samples (default: 7500).
        Otherwise, it uses the entire train dataset.

        Args:
            max_samples (int): Number of samples to use for preprocessing if the train dataset is large.
        """
        if len(self.x_train) > preprocessing_size:
            self.x_preprocessing = self.x_train.sample(
                n=preprocessing_size, random_state=random_state
            )
            self.y_preprocessing = self.y_train.loc[self.x_preprocessing.index]
            print(f"Preprocessing dataset created with {preprocessing_size} samples.")
        else:
            self.x_preprocessing = self.x_train
            self.y_preprocessing = self.y_train
            print("Preprocessing dataset uses the entire train dataset.")

    def describe(self) -> None:
        size = self.general_data.shape[0]
        features = self.x_train.shape[1]
        size_train = self.y_train.shape[0]
        size_test = self.y_test.shape[0]
        size_preprocessing = self.y_preprocessing.shape[0]
        print(
            f"\nDataset size: {size}\nNumber of features: {features}\nTrain subset size: {size_train}\nTest subset size: {size_test}"
        )
        if size_preprocessing != size_train:
            print(
                "This dataset contains a preprocessing subset of 7500 samples for hyperparameter optimization and feature selection"
            )

    @staticmethod
    def load_dataset(file_path) -> DatasetWrapper:
        instance = DatasetWrapper()
        instance.load(file_path=file_path)
        return instance

    @staticmethod
    def load_unsplit_dataset(
        full_df,
        target_column,
        nonfeature_columns,
        holdout_size,
        random_state,
        preprocessing_size,
    ) -> DatasetWrapper:
        instance = DatasetWrapper()
        instance.load_unsplit_dataframe(
            full_df=full_df,
            target_column=target_column,
            nonfeature_columns=nonfeature_columns,
            holdout_size=holdout_size,
            random_state=random_state,
            preprocessing_size=preprocessing_size,
        )
        return instance


def get_activity_data(
    target_chembl_id: str, activity_type: str, convert_units: bool = True
) -> pd.DataFrame:
    """
    Fetches and processes activity data from the ChEMBL database for a given target ID and activity type.
    Args:
        target_chembl_id (str): The ChEMBL ID of the target.
        activity_type (str): The type of activity to filter results by.
        convert_units (int): Whether to convert units to mol/L (1 for yes, 0 for no).
    Returns:
        dataset (pd.DataFrame): A dataframe object containing the processed dataset.
    """
    activity = new_client.activity
    activity_query = activity.filter(target_chembl_id=target_chembl_id)
    activity_query = activity_query.filter(standard_type=activity_type)
    activity_df: pd.DataFrame = pd.DataFrame(data=activity_query)
    columns = [
        "canonical_smiles",
        "molecule_chembl_id",
        "standard_type",
        "standard_value",
        "standard_units",
        "assay_description",
    ]
    activity_df = activity_df[columns]
    activity_df["standard_value"] = pd.to_numeric(
        arg=activity_df["standard_value"], errors="coerce"
    )
    activity_df = activity_df[activity_df["standard_value"] > 0]
    activity_df = (
        activity_df.dropna().drop_duplicates("canonical_smiles").reset_index(drop=True)
    )
    activity_df = mmm.getLipinskiDescriptors(molecules_df=activity_df)
    activity_df = mmm.getRo5Violations(molecules_df=activity_df)
    if convert_units:
        activity_df = mmm.convert_to_M(molecules_df=activity_df)
    activity_df = mm.normalizeValue(molecules_df=activity_df)
    activity_df = mm.getNegLog(molecules_df=activity_df)
    bioactivity_class = []

    for i in activity_df.standard_value:
        if float(i) >= (0.00001):  # 10000 nmol/L
            bioactivity_class.append("inactive")
        elif float(i) < (0.000001):  # 1000 mol/L
            bioactivity_class.append("active")
        else:
            bioactivity_class.append("intermediate")

    activity_df["bioactivity_class"] = bioactivity_class
    return activity_df


def calculate_fingerprint(
    activity_df: pd.DataFrame,
    fingerprint: str | list[str] = "pubchem",
) -> pd.DataFrame:

    fingerprint_dict = {
        "atompairs2d": "fingerprint/AtomPairs2DFingerprinter.xml",
        "atompairs2dcount": "fingerprint/AtomPairs2DFingerprintCount.xml",
        "estate": "fingerprint/EStateFingerprinter.xml",
        "extended": "fingerprint/ExtendedFingerprinter.xml",
        "fingerprinter": "fingerprint/Fingerprinter.xml",
        "graphonly": "fingerprint/GraphOnlyFingerprinter.xml",
        "klekota": "fingerprint/KlekotaRothFingerprinter.xml",
        "klekotacount": "fingerprint/KlekotaRothFingerprintCount.xml",
        "maccs": "fingerprint/MACCSFingerprinter.xml",
        "pubchem": "fingerprint/PubchemFingerprinter.xml",
        "substructure": "fingerprint/SubstructureFingerprinter.xml",
        "substructurecount": "fingerprint/SubstructureFingerprintCount.xml",
    }

    if type(fingerprint) == str:
        fingerprint = [fingerprint]

    descriptors_df = pd.DataFrame(index=activity_df.index)

    for i in fingerprint:
        fingerprint_path = fingerprint_dict[fingerprint]
        mmm.calculate_fingerprint(dataframe=activity_df, fingerprint=fingerprint_path)
        descriptors_df_i = pd.read_csv("descriptors.csv")
        descriptors_df_i = descriptors_df_i.drop("Name", axis=1)
        descriptors_df_i = pd.DataFrame(data=descriptors_df_i, index=activity_df.index)
        descriptors_df = pd.concat(objs=[descriptors_df, descriptors_df_i], axis=1)
        os.remove("descriptors.csv")
        os.remove("descriptors.csv.log")
    descriptors_df = mlm.scale_features(
        features=descriptors_df, scaler=preproc.MinMaxScaler()
    )
    descriptors_df = mm.remove_low_variance_columns(
        input_data=descriptors_df, threshold=0
    )

    return descriptors_df


def assemble_dataset(
    activity_df,
    descriptors_df,
    target_column="neg_log_value",
    holdout_size=0.2,
    random_state=42,
    preprocessing_size=7500,
) -> DatasetWrapper:
    nonfeature_columns = activity_df.columns
    dataset_df = pd.concat([activity_df, descriptors_df], axis=1)
    dataset_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    dataset_df.dropna(inplace=True)
    dataset = DatasetWrapper.load_unsplit_dataset(
        full_df=dataset_df,
        target_column=target_column,
        nonfeature_columns=nonfeature_columns,
        holdout_size=holdout_size,
        random_state=random_state,
        preprocessing_size=preprocessing_size,
    )

    return dataset


class MLConfig:

    def __init__(
        self, algorithm_name=None, algorithm=None, scoring=None, param_grid=None
    ):
        self.algorithm_name = algorithm_name
        self.algorithm = algorithm
        self.scoring = scoring
        self.param_grid = param_grid
        self.params = {}

    def set_scoring(self, scoring_list: str | list[str], alpha: float) -> None:
        scoring_dict: dict = {
            "r2": metrics.make_scorer(metrics.r2_score),
            "rmse": metrics.make_scorer(
                lambda y_true, y_pred: metrics.root_mean_squared_error(y_true, y_pred)
            ),
            "mae": metrics.make_scorer(metrics.mean_absolute_error),
            "quantile": metrics.make_scorer(
                lambda y_true, y_pred: metrics.mean_pinball_loss(
                    y_true, y_pred, alpha=alpha
                )
            ),
        }
        if type(scoring_list) == str:
            scoring_list = [scoring_list]
        scoring: dict = {}
        for i in scoring_list:
            try:
                metric = scoring_dict[i]
                scoring[i] = metric
            except KeyError as e:
                print(f"{i} is not available as a scoring metric")
                print(e)
        #TODO: add logic to add external scoring callables
        self.scoring = scoring

    def set_algorithm(
        self,
        algorithm_name: str | None,
        algorithm,
        random_state,
    ) -> None:

        algorithms: dict = {
            "bagging_reg": BaggingRegressor(random_state=random_state),
            "extratrees_reg": ExtraTreesRegressor(random_state=random_state),
            "gradboost_reg": GradientBoostingRegressor(random_state=random_state),
            "histgradboost_reg": HistGradientBoostingRegressor(
                random_state=random_state
            ),
            "randomforest_reg": RandomForestRegressor(random_state=random_state),
            "xgboost_reg": XGBRegressor(random_state=random_state),
        }
        if algorithm_name == None:
            print("algorithm_name not provided")
            if algorithm == None:
                print("please provide an algorithm object")
            print(
                "package functionality might not work properly with external algorithms"
            )
            self.algorithm = algorithm
        else:
            try:
                self.algorithm = algorithm[algorithm_name]
                self.param_grid = param_grids[algorithm_name]
            except KeyError as e:
                print("provided algorithm_name not available, check docs")
                print(e)

    def optimize_hyperparameters(
        self,
        dataset: DatasetWrapper,
        param_grid: dict | None,
        population_size=30,
        generations=30,
        refit="r2",
    ) -> None:
        if self.algorithm == None:
            print("define algorithm using setup()")
            return
        if dataset.x_train == None:
            print("dataset empty")
            return
        if (self.algorithm_name == None) and (param_grid == None):
            print(
                "param_grid not provided. provide a param_grid compatible with sklearn_genetic (https://sklearn-genetic-opt.readthedocs.io/en/stable/api/space.html)\nalternatively, provide algorithm_name, to use one of the param_grids provided"
            )
        param_grids: dict = {
            "bagging_reg": {
                "n_estimators": Integer(lower=10, upper=1000),  # 10
                "max_samples": Continuous(lower=0.1, upper=1.0),  # 1.0
                "max_features": Continuous(lower=0.1, upper=1.0),  # 1.0
                "bootstrap": Categorical(
                    choices=[True, False]
                ),  # Whether samples are drawn with replacement
                "bootstrap_features": Categorical(
                    choices=[True, False]
                ),  # Whether features are drawn with replacement
            },
            "extratrees_reg": {
                "n_estimators": Integer(lower=100, upper=2000),  # 100
                "max_depth": Categorical([None]),  # None
                "min_samples_split": Integer(lower=2, upper=20),  # 2
                "min_samples_leaf": Integer(lower=1, upper=20),  # 1
                "max_features": Continuous(
                    lower=0.1, upper=1
                ),  # Number of features to consider for splits
                "bootstrap": Categorical(
                    choices=[True, False]
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
                    choices=["squared_error", "absolute_error"]
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
                    lower=0.1, upper=1
                ),  # Fraction of samples used for fitting
                "colsample_bytree": Continuous(
                    lower=0.1, upper=1
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
                    choices=[True, False]
                ),  # Whether bootstrap samples are used
            },
            "xgboost_reg": {
                "n_estimators": Integer(lower=100, upper=2000),
                "learning_rate": Continuous(lower=0.001, upper=1),
                "max_depth": Integer(lower=0, upper=100),
                "min_child_weight": Continuous(
                    lower=0.1, upper=2.0
                ),  # Minimum sum of instance weight needed in a child
                "gamma": Continuous(
                    lower=0.1, upper=2.0
                ),  # Minimum loss reduction required to make a split
                "subsample": Continuous(
                    lower=0.1, upper=1
                ),  # Fraction of samples used for fitting
                "colsample_bytree": Continuous(
                    lower=0.1, upper=1
                ),  # Fraction of features used for fitting
                "reg_alpha": Continuous(lower=0.1, upper=2.0),  # L1 regularization
                "reg_lambda": Continuous(lower=0.1, upper=2.0),  # L2 regularization
            },
        }
        if param_grid == None:
            try:
                param_grid = param_grids[algorithm_name]
            except KeyError as e:
                print("provided algorithm_name not available, check docs")
        try:
            search_cv_results, best_params, time_to_execute = mlm.evaluate_and_optimize(
                algorithm=self.algorithm,
                param_grid=param_grid,
                x_train=dataset.x_train,
                y_train=dataset.y_train,
                scoring=self.scoring,
                algorithm_name=self.algorithm_name,
                population_size=population_size,
                generations=generations,
                refit=refit,
            )
        except Exception as e:
            print(f"something went wrong, check error:\n\n{e}")
        self.params = best_params
        return search_cv_results, best_params, time_to_execute

    @staticmethod
    def setup(
        algorithm_name=None,
        scoring_list: str | list[str] = ["r2"],
        random_state: int = 42,
        algorithm=None,
        **scoring_params,
    ) -> MLConfig:
        instance = MLConfig()
        instance.set_algorithm(
            algorithm_name=algorithm_name,
            algorithm=algorithm,
            random_state=random_state,
        )

        alpha: float = 0.5
        if "alpha" in scoring_params.keys():
            if not 0 < scoring_params["alpha"] <= 1:
                print("alpha must be between 0 and 1")
            try:
                alpha = scoring_params["alpha"]
            except TypeError as e:
                print("provided alpha not a valid float number")

        instance.set_scoring(scoring_list=scoring_list, alpha=alpha)


def residue_diagnosis():

    return


def deploy():

    return


# algoritmo
# tipo de erro
# classificação x regressão
# otimização
# implementação em dataset externo
# diagnóstico de resíduos
