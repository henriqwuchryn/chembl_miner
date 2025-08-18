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

    def subset_general_data(self, subset="train"):
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

    def save(self, file_path):
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

    def load(self, file_path):
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
    ):
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

    def create_preprocessing_dataset(self, random_state, preprocessing_size=7500):
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

    def describe(self):
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
    def load_dataset(file_path):
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
    ):
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

    def __init__(self, task=None, algorithm=None, scoring=None):
        self.task = task
        self.algorithm = algorithm
        self.scoring = scoring

    def set_scoring(self, scoring_list: str | list[str], quantile_alpha=0.9):
        scoring_dict: dict = {
            "r2": metrics.make_scorer(metrics.r2_score),
            "rmse": metrics.make_scorer(
                lambda y_true, y_pred: metrics.root_mean_squared_error(y_true, y_pred)
            ),
            "mae": metrics.make_scorer(metrics.mean_absolute_error),
            "quantile": metrics.make_scorer(
                lambda y_true, y_pred: metrics.mean_pinball_loss(
                    y_true, y_pred, alpha=quantile_alpha
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
        self.scoring = scoring
        return scoring


# algoritmo
# tipo de erro
# classificação x regressão
# otimização
# implementação em dataset externo
# diagnóstico de resíduos
