import os

import numpy as np
import pandas as pd
from chembl_webresource_client.new_client import new_client
from sklearn.base import BaseEstimator
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_pinball_loss
from sklearn.metrics import r2_score
from sklearn.metrics import root_mean_squared_error  # type: ignore
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn_genetic import GASearchCV
from sklearn_genetic.callbacks import DeltaThreshold
from sklearn_genetic.space import Categorical
from sklearn_genetic.space import Continuous
from sklearn_genetic.space import Integer
from xgboost import XGBRegressor

import machine_learning_methods as mlm
import miscelanneous_methods as mm
import molecules_manipulation_methods as mmm


def _print(input_string: str, verbosity: bool) -> None:
    if verbosity:
        print(input_string)


class DatasetWrapper:

    def __init__(
        self,
        general_data=pd.DataFrame(),
        x_train=pd.DataFrame(),
        x_test=pd.DataFrame(),
        y_train=pd.DataFrame(),
        y_test=pd.DataFrame(),
        x_preprocessing=pd.DataFrame(),
        y_preprocessing=pd.DataFrame(),
        ):
        """
        Initializes the DatasetWrapper with optional dataframes.
        """
        self.general_data = general_data
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.x_preprocessing = x_preprocessing
        self.y_preprocessing = y_preprocessing


    @classmethod
    def from_path(cls, file_path):
        instance = cls()
        instance.load(file_path=file_path)
        return instance


    @classmethod
    def from_dataframe(
        cls,
        full_df: pd.DataFrame,
        target_column: str,
        nonfeature_columns,
        use_structural_split: bool,
        holdout_size: float,
        random_state: int,
        preprocessing_size: int,
        ):
        instance = cls()
        instance._load_unsplit_dataframe(
            full_df=full_df,
            target_column=target_column,
            nonfeature_columns=nonfeature_columns,
            use_structural_split=use_structural_split,
            holdout_size=holdout_size,
            random_state=random_state,
            preprocessing_size=preprocessing_size,
            )
        return instance


    def subset_general_data(self, train_subset: bool = True) -> pd.DataFrame:
        """
        Retrieves the corresponding rows from general_data for the train or test dataset.

        Args:
            train_subset (bool): Indicates whether to subset train or test values. Default value is True.

        Returns:
            pd.DataFrame: Rows from general_data corresponding to the specified subset.
        """
        if train_subset:
            return self.general_data.loc[self.x_train.index]
        else:
            return self.general_data.loc[self.x_test.index]


    def save(self, file_path) -> None:
        """
        Saves the entire dataset to CSV files inside file_path folder.
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
        print(f"Dataset saved to {file_path} folder")


    def load(self, file_path) -> None:
        """
        Loads the dataset from CSV files inside file_path folder.
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
                f"{file_path}/fpr.csv",
                index_col="index",
                )
            self.y_preprocessing = pd.read_csv(
                f"{file_path}/tpr.csv",
                index_col="index",
                )["neg_log_value"]
            print(f"Dataset loaded from {file_path}")
        except Exception as e:
            print(e)
            print("Dataset loading failed")


    def _load_unsplit_dataframe(
        self,
        full_df: pd.DataFrame,
        target_column: str,
        nonfeature_columns,
        use_structural_split: bool,
        holdout_size: float,
        random_state: int,
        preprocessing_size: int,
        ) -> None:
        """
        Loads an unsplit dataframe containing all columns and splits it into
        general_data, x_train/test, and y_train/test.

        Args:
            full_df (pandas.DataFrame): Unsplit dataframe containing the dataset.
            target_column (str): Column name representing the target variable.
            nonfeature_columns: Columns representing nonfeature variables.
            use_structural_split (bool): Whether to use structural splitting or not.
            holdout_size (float): Proportion of the dataset to include in the test split. Standard value = 0.2.
            random_state (int): Random state for reproducibility. Standard value = 42.
            preprocessing_size (int): Maximum size of preprocessing dataset.
        """
        self.general_data = full_df[nonfeature_columns]
        features = full_df.drop(columns=nonfeature_columns)
        target = full_df[target_column]
        if use_structural_split:
            train_index, test_index = mlm.scaffold_split(
                activity_df=self.general_data,
                test_size=holdout_size,
                )
            self.x_train = features.loc[train_index]
            self.x_test = features.loc[test_index]
            self.y_train = target.loc[train_index]
            self.y_test = target.loc[test_index]
        else:
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
                features,
                target,
                train_size=holdout_size,
                random_state=random_state,
                )
        print(f"Unsplit dataset loaded and split into train/test sets")
        # Create preprocessing dataset
        self._create_preprocessing_dataset(
            random_state=random_state,
            preprocessing_size=preprocessing_size,
            )


    def _create_preprocessing_dataset(
        self,
        random_state,
        preprocessing_size=7500,
        ) -> None:
        """
        Creates a preprocessing dataset by sampling from the train dataset.
        If the train dataset is larger max_samples, it samples this amount of samples (default: 7500).
        Otherwise, it uses the entire train dataset.

        Args:
            random_state (int): Random state to use for splitting the dataset.
            preprocessing_size (int): Number of samples to use for preprocessing if the train dataset is large.
        """
        if len(self.x_train) > preprocessing_size:
            self.x_preprocessing = self.x_train.sample(
                n=preprocessing_size,
                random_state=random_state,
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
            f"\nDataset size: {size}\nNumber of features: {features}\nTrain subset size: {size_train}\nTest subset size: {size_test}",
            )
        if size_preprocessing != size_train:
            print(
                "This dataset contains a preprocessing subset of 7500 samples for hyperparameter optimization and feature selection",
                )



def get_activity_data(
    target_chembl_id: str,
    activity_type: str,
    ) -> pd.DataFrame:
    """
    Fetches and processes activity data from the ChEMBL database for a given target ID and activity type.
    Args:
        target_chembl_id (str): The ChEMBL ID of the target.
        activity_type (str): The type of activity to filter results by.
    Returns:
        dataset (pd.DataFrame): A dataframe object containing the processed dataset.
    """
    activity = new_client.activity  # type: ignore
    activity_query = activity.filter(target_chembl_id=target_chembl_id)
    activity_query = activity_query.filter(standard_type=activity_type)
    activity_df: pd.DataFrame = pd.DataFrame(data=activity_query)
    columns = [
        "molecule_chembl_id",
        "canonical_smiles",
        "molecule_pref_name",
        "target_chembl_id",
        "target_pref_name",
        "assay_chembl_id",
        "assay_description",
        "standard_type",
        "standard_value",
        "standard_units",
        ]
    activity_df = activity_df[columns]

    return activity_df


def review_assays(
    activity_df: pd.DataFrame,
    max_entries: int = 20,
    assay_keywords: list[str] | None = None,
    exclude_keywords: bool = False,
    inner_join: bool = False,
    ) -> list[str] | None:
    """
    Displays and filter assays from an activity DataFrame.

    Can either include or exclude assays based on keywords found in the
    'assay_description' column.

    Args:
        activity_df (pd.DataFrame): DataFrame containing activity data with
                                    "assay_chembl_id" and "assay_description" columns.
        max_entries (int): The number of top assays to display. Defaults to 20.
        assay_keywords (list[str] | None): A list of keywords to filter
                                           assays by. Defaults to None.
        exclude_keywords (bool):Use the keywords for exclusion instead of inclusion.
                                Defaults to False.
        inner_join (bool): Use inner join instead of outer (AND instead of OR).
                           Defaults to False.

    Returns:
        List[str] | None: A list of selected assay ChEMBL IDs, or None if no
                          keywords are provided or an error occurs.
    """
    assay_info = activity_df.loc[:, ["assay_chembl_id", "assay_description"]]
    unique_assays = len(assay_info.value_counts())
    print(
        f"Displaying {min(unique_assays, max_entries)} of {unique_assays} total unique assays.",
        )
    print("To see more, adjust the 'max_entries' parameter.\n")
    pd.set_option("display.max_rows", max_entries)
    print(assay_info.value_counts().head(n=max_entries))

    if assay_keywords is None:
        return None
    else:
        if inner_join:
            pattern = "".join([rf"(?=.*{keyword})" for keyword in assay_keywords])
        else:
            pattern = "|".join(assay_keywords)
        print(f"\nfiltering assays by: {assay_keywords}")
        print(f"exclude keywords: {exclude_keywords}")
        print(f"inner join : {inner_join}")

        mask = assay_info.loc[:, "assay_description"].str.contains(
            pattern,
            case=False,
            na=False,
            )
        if exclude_keywords:
            selected_assays = assay_info[~mask]
        else:
            selected_assays = assay_info[mask]
        unique_selected_assays = len(selected_assays.value_counts())
        print(
            f"Displaying {min(unique_selected_assays, max_entries)} of {unique_selected_assays} filtered assays.\n",
            )
        print(selected_assays.value_counts().head(n=max_entries))
        selected_id_list = selected_assays.loc[:, "assay_chembl_id"].unique().tolist()  # type: ignore
        return selected_id_list


def filter_by_assay(
    activity_df: pd.DataFrame,
    assay_ids: list[str],
    ) -> pd.DataFrame:
    """
    Filters an activity DataFrame by 'assay_description' column using provided assay_ids.
    Args:
        activity_df (pd.DataFrame): DataFrame containing ChEMBL activity data.
        assay_ids (list[str]): list of assay_chembl_ids obtained from the review_assays function.

    Returns:
        pd.DataFrame:
    """
    print(f"dataframe initial size: {activity_df.shape[0]}")
    filtered_activity_df = activity_df.loc[
        activity_df["assay_chembl_id"].isin(assay_ids)
    ]
    print(f"dataframe filtered size: {filtered_activity_df.shape[0]}")
    if filtered_activity_df.empty:
        print("filtration emptied dataframe, returning original dataframe")
        return activity_df
    else:
        return filtered_activity_df


def preprocess_data(
    activity_df: pd.DataFrame,
    convert_units: bool = True,
    assay_ids: list[str] | None = None,
    duplicate_treatment="median",
    activity_thresholds: dict[str, float] | None = {
        "active"      : 1000,
        "intermediate": 10000,
        },
    ) -> pd.DataFrame:
    activity_df["standard_value"] = pd.to_numeric(
        arg=activity_df["standard_value"],
        errors="coerce",
        )
    activity_df = activity_df[activity_df["standard_value"] > 0]
    activity_df = activity_df.dropna(subset=["molecule_chembl_id", "canonical_smiles", "standard_value"])
    if assay_ids is not None:
        activity_df = filter_by_assay(activity_df=activity_df, assay_ids=assay_ids)
    activity_df = mmm.get_lipinski_descriptors(molecules_df=activity_df)
    activity_df = mmm.get_ro5_violations(molecules_df=activity_df)
    if convert_units:
        activity_df = mmm.convert_to_m(molecules_df=activity_df)
    activity_df = mm.treat_duplicates(
        molecules_df=activity_df,
        method=duplicate_treatment,
        )
    activity_df = mm.normalize_value(molecules_df=activity_df)
    activity_df = mm.get_neg_log(molecules_df=activity_df)
    activity_df = activity_df.reset_index(drop=True)
    if activity_thresholds is not None:
        bioactivity_class = []
        sorted_thresholds = sorted(
            activity_thresholds.items(),
            key=lambda item: item[1],
            )

        for i in activity_df.standard_value:
            value_nm = float(i) * 1e9  # mol/l to n mol/L
            assigned_class = "inactive"
            for class_name, threshold_nM in sorted_thresholds:
                if value_nm <= threshold_nM:
                    assigned_class = class_name
                    break
            bioactivity_class.append(assigned_class)

        activity_df["bioactivity_class"] = bioactivity_class

    return activity_df


def calculate_fingerprint(
    activity_df: pd.DataFrame,
    fingerprint: str | list[str] = "pubchem",
    ) -> pd.DataFrame:
    # TODO: generalizar para demais descritores
    fingerprint_dict = {
        "atompairs2d"      : "fingerprint/AtomPairs2DFingerprinter.xml",
        "atompairs2dcount" : "fingerprint/AtomPairs2DFingerprintCount.xml",
        "estate"           : "fingerprint/EStateFingerprinter.xml",
        "extended"         : "fingerprint/ExtendedFingerprinter.xml",
        "fingerprinter"    : "fingerprint/Fingerprinter.xml",
        "graphonly"        : "fingerprint/GraphOnlyFingerprinter.xml",
        "klekota"          : "fingerprint/KlekotaRothFingerprinter.xml",
        "klekotacount"     : "fingerprint/KlekotaRothFingerprintCount.xml",
        "maccs"            : "fingerprint/MACCSFingerprinter.xml",
        "pubchem"          : "fingerprint/PubchemFingerprinter.xml",
        "substructure"     : "fingerprint/SubstructureFingerprinter.xml",
        "substructurecount": "fingerprint/SubstructureFingerprintCount.xml",
        }

    if type(fingerprint) == str:
        fingerprint = [fingerprint]

    descriptors_df = pd.DataFrame(index=activity_df.index)

    for i in fingerprint:
        fingerprint_path = fingerprint_dict[i]
        mmm.calculate_fingerprint(dataframe=activity_df, fingerprint=fingerprint_path)
        descriptors_df_i = pd.read_csv("descriptors.csv")
        descriptors_df_i = descriptors_df_i.drop("Name", axis=1)
        descriptors_df_i = pd.DataFrame(data=descriptors_df_i, index=activity_df.index)
        descriptors_df = pd.concat(objs=[descriptors_df, descriptors_df_i], axis=1)
        os.remove("descriptors.csv")
        os.remove("descriptors.csv.log")
    descriptors_df = mm.remove_low_variance_columns(
        input_data=descriptors_df,
        threshold=0,
        )

    return descriptors_df


def assemble_dataset(
    activity_df: pd.DataFrame,
    descriptors_df: pd.DataFrame,
    target_column: str = "neg_log_value",
    use_structural_split: bool = True,
    holdout_size: float = 0.2,
    random_state: int = 42,
    preprocessing_size: int = 7500,
    ) -> DatasetWrapper:
    nonfeature_columns = activity_df.columns
    dataset_df = pd.concat([activity_df, descriptors_df], axis=1)
    dataset_df = dataset_df.replace([np.inf, -np.inf], np.nan)
    essential_cols = list(descriptors_df.columns) + [target_column, "canonical_smiles"]
    dataset_df = dataset_df.dropna(subset=essential_cols)
    dataset = DatasetWrapper.from_dataframe(
        full_df=dataset_df,
        target_column=target_column,
        nonfeature_columns=nonfeature_columns,
        use_structural_split=use_structural_split,
        holdout_size=holdout_size,
        random_state=random_state,
        preprocessing_size=preprocessing_size,
        )

    return dataset


class MLConfig:

    def __init__(
        self,
        algorithm_name: str | None = None,
        algorithm: BaseEstimator | None = None,
        scoring: dict | None = None,
        param_grid: dict = {},
        params: dict = {},
        ):
        self.algorithm_name = algorithm_name
        self.algorithm = algorithm
        self.scoring = scoring
        self.param_grid = param_grid
        self.params = params


    def set_scoring(
        self,
        scoring_names: str | list[str],
        scoring: dict | None,  # type: ignore
        alpha: float,
        ) -> None:

        if scoring is None:
            scoring_dict: dict = {
                "r2"      : make_scorer(r2_score),
                "rmse"    : make_scorer(
                    lambda y_true, y_pred: root_mean_squared_error(y_true, y_pred),
                    greater_is_better=False
                    ),
                "mae"     : make_scorer(mean_absolute_error,greater_is_better=False),
                "quantile": make_scorer(
                    lambda y_true, y_pred: mean_pinball_loss(
                        y_true,
                        y_pred,
                        alpha=alpha,
                        ),
                    greater_is_better=False
                    ),
                }
            scoring: dict = {}
            if type(scoring_names) == str:
                scoring_names = [scoring_names]
            for i in scoring_names:
                try:
                    metric = scoring_dict[i]
                    scoring[i] = metric
                except KeyError as e:
                    print(f"{i} is not available as a scoring metric")
                    print(e)
        else:
            print(
                "package functionality might not work properly with external scoring functions",
                )
        self.scoring = scoring


    def set_algorithm(
        self,
        algorithm_name: str | None,
        algorithm: BaseEstimator | None,
        random_state: int,
        ) -> None:

        if algorithm is None:
            if algorithm_name is None:
                print("please, provide an algorithm_name or algorithm object")
            algorithms: dict = {
                "bagging_reg"      : BaggingRegressor(random_state=random_state),
                "extratrees_reg"   : ExtraTreesRegressor(random_state=random_state),
                "gradboost_reg"    : GradientBoostingRegressor(random_state=random_state),
                "histgradboost_reg": HistGradientBoostingRegressor(
                    random_state=random_state,
                    ),
                "randomforest_reg" : RandomForestRegressor(random_state=random_state),
                "xgboost_reg"      : XGBRegressor(random_state=random_state),
                }
            try:
                self.algorithm = algorithms[algorithm_name]
                self.algorithm_name = algorithm_name
            except KeyError as e:
                print(f"{algorithm_name} algorithm is not available, check docs")
                print(e)
        else:
            self.algorithm = algorithm
            print(
                "package functionality might not work properly with external algorithms",
                )


    def optimize_hyperparameters(
        self,
        dataset: DatasetWrapper,
        param_grid: dict | None,
        refit: str,
        population_size: int = 30,
        generations: int = 30,
        n_jobs=-1,
        ):
        if self.algorithm is None:
            print("define algorithm using setup()")
            return None
        if dataset.x_train.empty:
            print("dataset empty")
            return None
        if (self.algorithm_name is None) and (param_grid is None):
            print(
                "param_grid not provided. provide a param_grid compatible with sklearn_genetic (https://sklearn-genetic-opt.readthedocs.io/en/stable/api/space.html)\nalternatively, provide algorithm_name, to use one of the param_grids provided",
                )
            return None

        if param_grid is None:
            param_grids: dict = {
                "bagging_reg"      : {
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
                "extratrees_reg"   : {
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
                "gradboost_reg"    : {
                    "n_estimators"     : Integer(lower=100, upper=2000),  # 100
                    "learning_rate"    : Continuous(lower=0.001, upper=1),  # 0.1
                    "max_depth"        : Integer(lower=3, upper=100),  # 3
                    "min_samples_split": Integer(lower=2, upper=20),  # 2
                    "min_samples_leaf" : Integer(lower=1, upper=20),  # 1
                    "subsample"        : Continuous(lower=0.1, upper=1.0),  # 1.0
                    "max_features"     : Continuous(lower=0.1, upper=1.0),  # 1.0
                    },
                "histgradboost_reg": {
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
                "lgbm_reg"         : {
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
                "randomforest_reg" : {
                    "n_estimators"     : Integer(lower=100, upper=2000),  # 100
                    "max_depth"        : Categorical([None]),  # None
                    "min_samples_split": Integer(lower=2, upper=20),  # 2
                    "min_samples_leaf" : Integer(lower=1, upper=20),  # 1
                    "max_features"     : Continuous(lower=0.1, upper=1),  # 1.0
                    "bootstrap"        : Categorical(
                        choices=[True, False],
                        ),  # Whether bootstrap samples are used
                    },
                "xgboost_reg"      : {
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
            try:
                param_grid = param_grids[self.algorithm_name]
            except KeyError as e:
                print(f"{e}\n\nprovided algorithm_name not available, check docs")
                return None
        try:
            callback = DeltaThreshold(threshold=0.001, generations=3)
            param_search = GASearchCV(
                estimator=self.algorithm,
                param_grid=param_grid,
                scoring=self.scoring,
                population_size=population_size,
                generations=generations,
                refit=refit,  # type: ignore
                n_jobs=n_jobs,
                return_train_score=True,
                )
            param_search.fit(dataset.x_train, dataset.y_train, callbacks=callback)
        except Exception as e:
            print(f"something went wrong, check error:\n\n{e}")
            return None
        self.params = param_search.best_params_
        return param_search


    def evaluate_model(
        self,
        dataset: DatasetWrapper,
        cv: int = 10,
        params: dict | None = None,
        n_jobs=-1,
        ):

        if self.algorithm is None:
            print("define algorithm using setup()")
            return None
        if dataset.x_train.empty:
            print("dataset empty")
            return None
        if params is not None:
            algorithm = self.algorithm.set_params(**params)
        elif self.params is not None:
            algorithm = self.algorithm.set_params(**self.params)
        else:
            algorithm = self.algorithm

        cv_results = cross_validate(
            estimator=algorithm,
            X=dataset.x_train,
            y=dataset.y_train,
            cv=cv,
            scoring=self.scoring,
            n_jobs=n_jobs,
            return_train_score=True,
            )
        # TODO: add a way to visualize cv results with generalization
        return cv_results


    @staticmethod
    def setup(
        algorithm_name: str | None = None,
        algorithm: BaseEstimator | None = None,
        scoring_names: str | list[str] = ["r2", "rmse", "mae"],
        scoring: dict | None = None,
        random_state: int = 42,
        **scoring_params,
        ):
        if (algorithm_name is None) and (algorithm is None):
            print(
                "you must select an algorithm_name from the docs or provide an algorithm object",
                )
            return None

        instance = MLConfig()
        instance.set_algorithm(
            algorithm_name=algorithm_name,
            algorithm=algorithm,
            random_state=random_state,
            )

        # checking scoring_params for embedded scoring functions
        if "alpha" in scoring_params.keys():
            alpha = scoring_params["alpha"]
            if type(alpha) != float:
                print("alpha must be a float between 0 and 1")
                return None
            elif not 0 < alpha < 1:
                print("alpha must be between 0 and 1")
                return None
        else:
            alpha: float = 0.5

        instance.set_scoring(scoring_names=scoring_names, scoring=scoring, alpha=alpha)

        return instance


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
