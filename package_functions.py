import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from chembl_webresource_client.new_client import new_client
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
        self, full_df, target_column='neg_log_value', test_size=0.2, random_state=42
    ):
        """
        Loads an unsplit dataframe containing all columns and splits it into
        general_data, x_train/test, and y_train/test.

        Args:
            dataframe (pandas.DataFrame): Unsplit dataframe containing the dataset.
            general_columns (list): List of columns representing general data.
            target_column (str): Column name representing the target variable.
            test_size (float): Proportion of the dataset to include in the test split. Standard value = 0.2.
            random_state (int): Random state for reproducibility. Standard value = 42.
        """
        general_columns = [
            "canonical_smiles",
            "molecule_chembl_id",
            "standard_type",
            "standard_value",
            "standard_units",
            "assay_description",
            "neg_log_value",
            "lipinski_descriptors",
            "ro5_violations",
            "bioactivity_class",
        ]
        self.general_data = full_df[general_columns]
        features = full_df.drop(columns=general_columns)
        target = full_df[target_column]
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            features, target, test_size=test_size, random_state=random_state
        )
        self.file_path = file_path
        print(f"Unsplit dataset loaded and split into train/test sets")
        # Create preprocessing dataset
        self.create_preprocessing_dataset()

    def create_preprocessing_dataset(self, max_samples=7500):
        """
        Creates a preprocessing dataset by sampling from the train dataset.
        If the train dataset is larger than 10,000 samples, it samples 7,500 examples.
        Otherwise, it uses the entire train dataset.

        Args:
            max_samples (int): Number of samples to use for preprocessing if the train dataset is large.
        """
        if len(self.x_train) > 10000:
            self.x_preprocessing = self.x_train.sample(n=max_samples, random_state=42)
            self.y_preprocessing = self.y_train.loc[self.x_preprocessing.index]
            print(f"Preprocessing dataset created with {max_samples} samples.")
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
        instance.load(file_path)
        return instance

    @staticmethod
    def load_unsplit_dataset(
        full_df, target_column, test_size=0.2, random_state=42
    ):
        instance = DatasetWrapper()
        instance.load_unsplit_dataframe(
            full_df, target_column, test_size=0.2, random_state=42
        )
        return instance


def get_activity_data(
    target_chembl_id: str, activity_type: str, convert_units: int = 1, target_column='neg_log_value'
) -> pd.DataFrame:
    """
    Fetches and processes activity data from the ChEMBL database for a given target ID and activity type.
    Args:
        target_chembl_id (str): The ChEMBL ID of the target.
        activity_type (str): The type of activity to filter results by.
        convert_units (int): Whether to convert units to mol/L (1 for yes, 0 for no).
    Returns:
        dataset (DatasetWrapper): A DatasetWrapper object containing the processed dataset.
    """
    activity = new_client.activity
    activity_query = activity.filter(target_chembl_id=target_chembl_id)
    activity_query = activity_query.filter(standard_type=activity_type)
    activity_df: pd.DataFrame = pd.DataFrame(activity_query)
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
        activity_df["standard_value"], errors="coerce"
    )
    activity_df = activity_df[activity_df["standard_value"] > 0]
    activity_df = (
        activity_df.dropna().drop_duplicates("canonical_smiles").reset_index(drop=True)
    )
    activity_df = mmm.getLipinskiDescriptors(activity_df)
    activity_df = mmm.getRo5Violations(activity_df)
    if convert_units == 1:
        activity_df = mmm.convert_to_M(activity_df)
    activity_df = mm.normalizeValue(activity_df)
    activity_df = mm.getNegLog(activity_df)
    bioactivity_class = []

    for i in activity_df.standard_value:
        if float(i) >= (0.00001):  # 10000 nmol/L
            bioactivity_class.append("inactive")
        elif float(i) < (0.000001):  # 1000 mol/L
            bioactivity_class.append("active")
        else:
            bioactivity_class.append("intermediate")

    activity_df["bioactivity_class"] = bioactivity_class
    dataset = DatasetWrapper.load_unsplit_dataset(activity_df, target_column)
    return dataset
    # ADD IMPLEMENTATION TO DATASET WRAPPER
