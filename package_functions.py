import os

import numpy as np
import pandas as pd
from chembl_webresource_client.new_client import new_client
from padelpy import padeldescriptor
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
from sklearn.metrics._scorer import _BaseScorer
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


# TODO: fix imports everywhere; migrate functions from another files here with leading underscore


verbosity: int = 1


def set_verbosity(verbosity_level: int) -> None:
    """Determines verbosity level.
    Verbosity of 0 means no output at all, except for erros.
    Verbosity of 1 means basic output containing information of each step start and end
    Verbosity of 2 means complete output, containing every information and parameter - Great for logs"""
    global verbosity
    if 0 <= verbosity_level <= 2:
        verbosity = verbosity_level
    else:
        print(f"Verbosity level must be 0, 1 or 2. Got {verbosity_level}.")
        print(f"Verbosity level is {verbosity}.")


def print_low(input_string) -> None:
    global verbosity
    if 1 <= verbosity <= 2:
        print(input_string)


def print_high(input_string) -> None:
    global verbosity
    if verbosity == 2:
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
        try:
            print_low(f"Loading DatasetWrapper object from {file_path}")
            instance._load_from_path(file_path=file_path)
            print_low(f"DatasetWrapper object loaded from {file_path}")
            print_high(f"Dataset size: {instance.general_data.shape[0]}")
            print_high(f"Train subset size: {len(instance.y_train)}")
            print_high(f"Test shape: {len(instance.y_test)}")
            print_high(f"Number of features: {instance.x_test.shape[1]}")
        except Exception as e:
            print("Dataset loading failed")
            raise e
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
        ):
        instance = cls()
        print_low("Loading DatasetWrapper object from unsplit dataframe and splitting data.")
        print_high(f"Target column: '{target_column}'")
        print_high(f"Holdout size: {holdout_size}")
        print_high(f"Using structural split: {use_structural_split}")
        print_high(f"Random state: {random_state}")
        try:
            instance._load_unsplit_dataframe(
                full_df=full_df,
                target_column=target_column,
                nonfeature_columns=nonfeature_columns,
                use_structural_split=use_structural_split,
                holdout_size=holdout_size,
                random_state=random_state,
                )
        except Exception as e:
            print("Dataset loading failed")
            raise e
        print_low(f"DatasetWrapper object loaded from unsplit DataFrame and split into train/test sets")
        print_high(f"Dataset size: {instance.general_data.shape[0]}")
        print_high(f"Train subset size: {len(instance.y_train)}")
        print_high(f"Test shape: {len(instance.y_test)}")
        print_high(f"Number of features: {instance.x_test.shape[1]}")
        return instance


    def subset_general_data(self, train_subset: bool = True) -> pd.DataFrame:
        """
        Retrieves the corresponding rows from general_data for the train or test dataset.

        Args:
            train_subset (bool): Indicates whether to subset train or test values. Default value is True.

        Returns:
            pd.DataFrame: Rows from general_data corresponding to the specified subset.
        """
        subset_type = "train" if train_subset else "test"
        print_high(f"Subsetting general_data for the {subset_type} set.")
        if train_subset:
            return self.general_data.loc[self.x_train.index]
        else:
            return self.general_data.loc[self.x_test.index]


    def to_path(self, file_path) -> None:
        """
        Saves the entire dataset to CSV files inside file_path folder.
        """

        if not os.path.exists(file_path):
            os.makedirs(file_path)
            print_high(f"Creating directory: {file_path}")

        print_low(f"Saving dataset to {file_path} folder")
        self.general_data.to_csv(f"{file_path}/general_data.csv", index_label="index")
        print_high(f"Saved general_data to {file_path}/general_data.csv")
        self.x_train.to_csv(f"{file_path}/x_train.csv", index_label="index")
        print_high(f"Saved x_train to {file_path}/x_train.csv")
        self.x_test.to_csv(f"{file_path}/x_test.csv", index_label="index")
        print_high(f"Saved x_test to {file_path}/x_test.csv")
        self.y_train.to_csv(f"{file_path}/y_train.csv", index_label="index")
        print_high(f"Saved y_train to {file_path}/y_train.csv")
        self.y_test.to_csv(f"{file_path}/y_test.csv", index_label="index")
        print_high(f"Saved y_test to {file_path}/y_test.csv")
        print_low(f"Dataset saved to {file_path} folder")


    def _load_from_path(self, file_path) -> None:
        """
        Loads the dataset from CSV files inside file_path folder.
        """
        self.general_data = pd.read_csv(f"{file_path}/general_data.csv", index_col="index")
        self.x_train = pd.read_csv(f"{file_path}/x_train.csv", index_col="index")
        self.x_test = pd.read_csv(f"{file_path}/x_test.csv", index_col="index")
        self.y_train = pd.read_csv(f"{file_path}/y_train.csv", index_col="index")[
            "neg_log_value"
        ]
        self.y_test = pd.read_csv(f"{file_path}/y_test.csv", index_col="index")[
            "neg_log_value"
        ]


    def _load_unsplit_dataframe(
        self,
        full_df: pd.DataFrame,
        target_column: str,
        nonfeature_columns,
        use_structural_split: bool,
        holdout_size: float,
        random_state: int,
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
        # Create preprocessing dataset


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
            print_low(f"Preprocessing dataset created with {preprocessing_size} samples.")
        else:
            self.x_preprocessing = self.x_train
            self.y_preprocessing = self.y_train
            print_low("Preprocessing dataset uses the entire train dataset.")


    def describe(self) -> None:

        print(f"Dataset size: {self.general_data.shape[0]}")
        print(f"Train subset size: {len(self.y_train)}")
        print(f"Test shape: {len(self.y_test)}")
        print(f"Number of features: {self.x_test.shape[1]}")


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
    print_low(f"🧪 Fetching '{activity_type}' activity data from ChEMBL for target: {target_chembl_id}")
    activity = new_client.activity  # type: ignore
    activity_query = activity.filter(target_chembl_id=target_chembl_id)
    activity_query = activity_query.filter(standard_type=activity_type)
    activity_df: pd.DataFrame = pd.DataFrame(data=activity_query)
    print_high(f"Fetched {activity_df.shape[0]} records.")
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
    print_low("✅ Data fetched successfully.")
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
    print_low(
        f"Displaying {min(unique_assays, max_entries)} of {unique_assays} total unique assays.",
        )
    print_low("To see more, adjust the 'max_entries' parameter.\n")
    pd.set_option("display.max_rows", max_entries)
    print_low(assay_info.value_counts().head(n=max_entries))

    if assay_keywords is None:
        print_high("No assay_keywords provided, returning None.")
        if verbosity == 0:
            print('No keywords provided. Increase verbosity to review assays.')
        return None
    else:
        if inner_join:
            pattern = "".join([rf"(?=.*{keyword})" for keyword in assay_keywords])
        else:
            pattern = "|".join(assay_keywords)
        print_low("Filtering assays by keywords.")
        print_high(f"Keywords: {assay_keywords}")
        print_high(f"Exclude keywords: {exclude_keywords}")
        print_high(f"Inner join (AND logic): {inner_join}")
        print_high(f"Resulting regex patter: {pattern}")

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
        print_low(
            f"Displaying {min(unique_selected_assays, max_entries)} of {unique_selected_assays} filtered assays.\n",
            )
        print_low(selected_assays.value_counts().head(n=max_entries))
        selected_id_list = selected_assays.loc[:, "assay_chembl_id"].unique().tolist()  # type: ignore
        return selected_id_list


def _filter_by_assay(
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

    filtered_activity_df = activity_df.loc[
        activity_df["assay_chembl_id"].isin(assay_ids)
    ]
    if filtered_activity_df.empty:
        print("Filtration by assay ids emptied dataframe, returning original dataframe.")
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
    print_low("Starting data preprocessing.")
    print_high("Converting 'standard_value' column to numeric, coercing errors (failing to convert will result in NA).")
    activity_df["standard_value"] = pd.to_numeric(
        arg=activity_df["standard_value"],
        errors="coerce",
        )
    print_high("Filtering out 'standard_value' entries with infinite values.")
    activity_df = activity_df.replace([np.inf, -np.inf], np.nan)
    print_high("Filtering out non-positive 'standard_value' entries.")
    activity_df = activity_df[activity_df["standard_value"] > 0]
    print_high("Dropping rows with NA in key columns.")
    activity_df = activity_df.dropna(subset=["molecule_chembl_id", "canonical_smiles", "standard_value"])
    if assay_ids is not None:
        print_low("Filtering DataFrame by assay ids.")
        print_high(f"Dataframe initial size: {activity_df.shape[0]}")
        activity_df = _filter_by_assay(activity_df=activity_df, assay_ids=assay_ids)
        print_high(f"Dataframe filtered size: {activity_df.shape[0]}")
    print_high("Calculating Lipinski descriptors.")
    activity_df = mmm.get_lipinski_descriptors(molecules_df=activity_df)
    print_high("Calculating Rule of 5 violations.")
    activity_df = mmm.get_ro5_violations(molecules_df=activity_df)
    if convert_units:
        print_high("Converting standard units to Molar (mol/L)")
        activity_df = mmm.convert_to_m(molecules_df=activity_df)
    print_high(f"Treating duplicates using '{duplicate_treatment}' method.")
    activity_df = mm.treat_duplicates(
        molecules_df=activity_df,
        method=duplicate_treatment,
        )
    print_high("Normalizing 'standard_value'.")
    activity_df = mm.normalize_value(molecules_df=activity_df)
    print_high("Calculating negative logarithm of 'standard_value'.")
    activity_df = mm.get_neg_log(molecules_df=activity_df)
    print_high("Resetting index.")

    activity_df = activity_df.reset_index(drop=True)
    if activity_thresholds is not None:
        print_high("Assigning bioactivity classes based on thresholds.")
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
    print_low("Data preprocessing complete.")

    return activity_df


def _calculate_fingerprint(
    activity_df: pd.DataFrame,
    smiles_col="canonical_smiles",
    fingerprint: str | list[str] = "pubchem",
    ) -> pd.DataFrame:

    df_smi = activity_df[smiles_col]
    df_smi.to_csv('molecules.smi', sep='\t', index=False, header=False)
    padeldescriptor(
        mol_dir='molecules.smi',
        d_file='descriptors.csv',
        descriptortypes=fingerprint,
        detectaromaticity=True,
        standardizenitro=True,
        standardizetautomers=True,
        threads=-1,
        removesalt=True,
        log=True,
        fingerprints=True,
        )
    descriptors_df_i = pd.read_csv("descriptors.csv")
    descriptors_df_i = descriptors_df_i.drop("Name", axis=1)
    descriptors_df_i = pd.DataFrame(data=descriptors_df_i, index=activity_df.index)
    os.remove("descriptors.csv")
    os.remove("descriptors.csv.log")
    os.remove("molecules.smi")
    return descriptors_df_i


def calculate_fingerprint(
    activity_df: pd.DataFrame,
    smiles_col="canonical_smiles",
    fingerprint: str | list[str] = "pubchem",
    remove_low_variance: bool = False,
    low_variance_threshold: float = 0.0,
    ) -> pd.DataFrame:
    # TODO: generalizar para demais descritores
    print_low("Starting fingerprint calculation.")
    print_high("This will create temporary files in this folder: descriptors.csv; descriptors.csv.log and molecules.smi")
    fingerprint_dict = {
        "atompairs2d"      : "fingerprint/AtomPairs2Dfingerprinter.xml",
        "atompairs2dcount" : "fingerprint/AtomPairs2DfingerprintCount.xml",
        "estate"           : "fingerprint/EStatefingerprinter.xml",
        "extended"         : "fingerprint/Extendedfingerprinter.xml",
        "fingerprinter"    : "fingerprint/fingerprinter.xml",
        "graphonly"        : "fingerprint/GraphOnlyfingerprinter.xml",
        "klekota"          : "fingerprint/KlekotaRothfingerprinter.xml",
        "klekotacount"     : "fingerprint/KlekotaRothfingerprintCount.xml",
        "maccs"            : "fingerprint/MACCSfingerprinter.xml",
        "pubchem"          : "fingerprint/Pubchemfingerprinter.xml",
        "substructure"     : "fingerprint/Substructurefingerprinter.xml",
        "substructurecount": "fingerprint/SubstructurefingerprintCount.xml",
        }

    if type(fingerprint) == str:
        fingerprint = [fingerprint]

    descriptors_df = pd.DataFrame(index=activity_df.index)

    for i in fingerprint:
        print_high(f"Calculating '{i}' fingerprint.")
        fingerprint_path = fingerprint_dict[i]
        descriptors_df_i = _calculate_fingerprint(
            activity_df=activity_df,
            smiles_col=smiles_col,
            fingerprint=fingerprint_path,
            )
        descriptors_df = pd.concat(objs=[descriptors_df, descriptors_df_i], axis=1)
    print_high(f"Total features from fingerprints: {descriptors_df.shape[1]}")
    if remove_low_variance:
        print_low("Removing low variance features.")
        print_high(f"Variance threshold: {low_variance_threshold}")
        initial_cols = descriptors_df.shape[1]
        descriptors_df = mm.remove_low_variance_columns(
            input_data=descriptors_df,
            threshold=low_variance_threshold,
            )
        final_cols = descriptors_df.shape[1]
        print_high(f"Removed {initial_cols - final_cols} low variance columns. Kept {final_cols}.")
    print_low("Fingerprint calculation complete.")

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
    print_low("Assembling dataset.")
    nonfeature_columns = activity_df.columns
    dataset_df = pd.concat([activity_df, descriptors_df], axis=1)
    dataset = DatasetWrapper.from_dataframe(
        full_df=dataset_df,
        target_column=target_column,
        nonfeature_columns=nonfeature_columns,
        use_structural_split=use_structural_split,
        holdout_size=holdout_size,
        random_state=random_state,
        preprocessing_size=preprocessing_size,
        )
    print_low("Dataset assembled.")

    return dataset


class DeployDatasetWrapper:

    def __init__(
        self,
        deploy_data: pd.DataFrame = None,
        deploy_descriptors: pd.DataFrame = None,
        prediction: pd.DataFrame = None,
        ) -> None:
        self.deploy_data = pd.DataFrame() if deploy_data is None else deploy_data
        self.deploy_descriptors = pd.DataFrame() if deploy_descriptors is None else deploy_descriptors
        self.prediction = pd.DataFrame() if prediction is None else prediction


    @classmethod
    def prepare_dataset(
        cls,
        deploy_data: pd.DataFrame,
        model_features,
        smiles_col: str = 'canonical_smiles',
        fingerprint: str = 'pubchem',
        ):
        print_low("Preparing DeployDatasetWrapper object.")
        instance = cls()
        instance.deploy_data = deploy_data
        instance.prepare_deploy_dataset(model_features=model_features, smiles_col=smiles_col, fingerprint=fingerprint)
        print_low("DeployDatasetWrapper object prepared.")
        return instance


    @classmethod
    def from_path(cls, file_path):
        if not os.path.exists(file_path):
            print("Provided file_path does not exist")
        print_low(f"Loading DeployDatasetWrapper object from {file_path}.")
        instance = cls()
        instance.deploy_data = pd.read_csv(f"{file_path}/deploy_data.csv")
        instance.deploy_descriptors = pd.read_csv(f"{file_path}/deploy_descriptors.csv")
        instance.prediction = pd.read_csv(f"{file_path}/prediction.csv")
        print_low("DeploymentDatasetWrapper object with data, descriptors, and previous predictions loaded.")
        return instance


    def to_path(self, file_path):
        print_low(f"Saving DeploymentDatasetWrapper object to {file_path}.")
        if not os.path.exists(file_path):
            print_high(f"Creating directory: {file_path}")
            os.makedirs(file_path)

        self.deploy_data.to_csv(f"{file_path}/deploy_data.csv", index_label="index")
        self.deploy_descriptors.to_csv(f"{file_path}/deploy_descriptors.csv", index_label="index")
        self.prediction.to_csv(f"{file_path}/prediction.csv", index_label="index")
        print_low("DeploymentDatasetWrapper object with data, descriptors, and predictions saved.")


    def prepare_deploy_dataset(
        self,
        model_features,
        smiles_col: str,
        fingerprint: str,
        ):

        if smiles_col not in self.deploy_data.columns:
            raise ValueError("Could not find smiles column in provided data")
        if self.deploy_descriptors.empty:
            print_low("No descriptors found on DeploymentDatasetWrapper object, calculating now.")
            self.deploy_descriptors = calculate_fingerprint(
                activity_df=self.deploy_data,
                smiles_col=smiles_col,
                fingerprint=fingerprint,
                )
        else:
            print_low('Descriptors already calculated, skipping calculation')
            print_low("Aligning deployment features with model features...")
            print_high(f"Deployment descriptors shape before alignment: {self.deploy_descriptors.shape}")
        try:
            self.deploy_descriptors = self.deploy_descriptors.loc[:, model_features]
            print_high(f"Deployment descriptors shape after alignment: {self.deploy_descriptors.shape}")
        except KeyError as e:
            print("Failed to align with model features.")
            print_low(
                "Please, rerun prepare_deploy_dataset method from DeployDatasetWrapper instance with new model_features iterable.",
                )
            print_low("Tip: use feature_names_in_ attribute from the model, or x_train.columns attribute from the dataset wrapper.\n", )
            print_low(e)


class MLWrapper:

    def __init__(
        self,
        algorithm_name: str | None = None,
        algorithm: BaseEstimator | None = None,
        fit_model: BaseEstimator | None = None,
        scoring: dict = None,
        param_grid: dict = None,
        params: dict = None,
        ):
        self.algorithm_name = algorithm_name
        self.algorithm = algorithm
        self.fit_model = fit_model
        self.scoring = {} if scoring is None else scoring
        self.param_grid = {} if param_grid is None else param_grid
        self.params = {} if params is None else params


    @staticmethod
    def setup(
        algorithm: BaseEstimator | str,
        scoring: dict | str | list[str] = ["r2", "rmse", "mae"],
        random_state: int = 42,
        n_jobs: int = -1,
        **scoring_params,
        ):
        print_low("Setting up MLWrapper object.")
        instance = MLWrapper()
        instance._set_algorithm(
            algorithm=algorithm,
            random_state=random_state,
            n_jobs=n_jobs,
            )
        print_high(f"Algorithm set to: {instance.algorithm_name}")
        print_high(f"Random state: {random_state}, n_jobs: {n_jobs}")

        # checking scoring_params for embedded scoring function parameters
        alpha: float = 0.5
        try:
            if ("alpha" not in scoring_params.keys()) and ('quantile' in scoring):
                raise ValueError("Parameter alpha (quantile) not provided.")
            if "alpha" in scoring_params.keys():
                alpha = scoring_params["alpha"]
                try:
                    alpha = float(alpha)
                except ValueError as e:
                    print_low("Parameter alpha (quantile) could not be converted to float.")
                    print(e)
                if not 0 < alpha < 1:
                    raise ValueError(
                        "Parameter alpha (quantile) must be a float between 0 and 1,",
                        )
                else:
                    print_high(f"Using alpha={alpha} for quantile scoring.")
        except ValueError:
            print("Could not use provided alpha, using standard value: 0.5")
            alpha: float = 0.5

        instance._set_scoring(scoring=scoring, alpha=alpha)
        print_high(f"Scoring metrics set to: {list(instance.scoring.keys())}")
        print_low("✅ MLWrapper setup complete.")

        return instance


    # TODO: implementar outros métodos de busca (grid, random)
    def optimize_hyperparameters(
        self,
        dataset: DatasetWrapper,
        cv: int = 3,
        param_grid: dict | None = None,
        refit: str | bool = True,
        population_size: int = 30,
        generations: int = 30,
        n_jobs=-1,
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
            if self.algorithm_name not in available_param_grids.keys():
                raise ValueError(
                    'provided algorithm_name does not have a param_grid available.\nPlease, provide a param_grid compatible with sklearn_genetic (https://sklearn-genetic-opt.readthedocs.io/en/stable/api/space.html)',
                    )
            print_high(f"Using pre-defined parameter grid for '{self.algorithm_name}'.")
            param_grid = available_param_grids[self.algorithm_name]
        else:
            print_high("Using user-provided parameter grid.")
        try:
            callback = DeltaThreshold(threshold=0.001, generations=3)
            param_search = GASearchCV(
                estimator=self.algorithm,
                cv=cv,
                param_grid=param_grid,
                scoring=self.scoring,
                population_size=population_size,
                generations=generations,
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
            print_low("Check resulting param_search to determine best parameters, or rerun with refit method",)
            print(e)
        return param_search


    def evaluate_model(
        self,
        dataset: DatasetWrapper,
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
            print_low("No provided parameters, using standard algorithm")
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


    def fit(
        self,
        dataset: DatasetWrapper,
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
            print("No provided parameters, using standard algorithm")
            _algorithm = self.algorithm

        print_high(f"Final model parameters: {_algorithm.get_params()}")
        fit_model = _algorithm.fit(X=dataset.x_train, y=dataset.y_train)
        self.fit_model = fit_model
        print_low("Model fitting complete.")
        return fit_model


    def residue_diagnosis(self):

        return


    def deploy(
        self,
        deploy_dataset: DeployDatasetWrapper,
        ):
        print_low("🚢 Deploying model and making predictions...")
        # TODO:
        # dominio aplicabilidade

        if self.fit_model is None:
            print('Model not fit. Please use the .fit() method first.')
            return None
        if deploy_dataset.deploy_descriptors.empty:
            print(
                'Deployment dataset provided does not contain descriptors. Please use prepare_dataset() or prepare_deploy_dataset() methods.',
                )
            return None

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


    # TODO
    # função q gera relatório pdf ***
    # dalex EXPLAIN
    # ebook - machine learning e exai - conflito publicação

    def _set_algorithm(
        self,
        algorithm: str | BaseEstimator,
        random_state: int,
        n_jobs: int,
        ) -> None:

        if isinstance(algorithm, str):
            available_algorithms: dict = {
                "bagging_reg"      : BaggingRegressor(random_state=random_state, n_jobs=n_jobs),
                "extratrees_reg"   : ExtraTreesRegressor(random_state=random_state, n_jobs=n_jobs),
                "gradboost_reg"    : GradientBoostingRegressor(random_state=random_state),
                "histgradboost_reg": HistGradientBoostingRegressor(
                    random_state=random_state,
                    ),
                "randomforest_reg" : RandomForestRegressor(random_state=random_state, n_jobs=n_jobs),
                "xgboost_reg"      : XGBRegressor(random_state=random_state, n_jobs=n_jobs),
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
        if self.algorithm is None:
            raise ValueError("Algorithm not set. Define algorithm using setup()")
        if self.scoring == {}:
            raise ValueError("Scorers not set. Define scoring using setup()")
        # if check_params:
        #     if self.params == {}:
        #         raise ValueError("define params using setup()")

# classificação x regressão
# implementação em dataset externo com AD
# diagnóstico de resíduos
# FILTRAGEM POR SIMILARIDADE NA BUSCA DO DATASET
# exploração de dados - MW, 

# implementar em R
