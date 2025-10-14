import os

import pandas as pd
from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.DataStructs import BulkTanimotoSimilarity
from rdkit.ML.Cluster import Butina
from sklearn.model_selection import train_test_split

from .utils import print_low, print_high


class TrainingData:

    def __init__(
        self,
        general_data=pd.DataFrame(),
        x_train=pd.DataFrame(),
        x_test=pd.DataFrame(),
        y_train=pd.Series(),
        y_test=pd.Series(),
        ):
        """
        Initializes the DatasetWrapper with optional dataframes.
        """
        self.general_data = general_data
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test


    @classmethod
    def from_path(cls, file_path, target_col: str = "neg_log_value"):
        instance = cls()
        try:
            print_low(f"Loading DatasetWrapper object from {file_path}")
            instance._load_from_path(file_path=file_path, target_col=target_col)
            print_low(f"DatasetWrapper object loaded from {file_path}")
            print_high(f"Dataset size: {instance.general_data.shape[0]}")
            print_high(f"Train subset size: {len(instance.y_train)}")
            print_high(f"Test subset size: {len(instance.y_test)}")
            print_high(f"Number of features: {instance.x_test.shape[1]}")
        except Exception as e:
            print("Dataset loading failed")
            raise e
        return instance


    @classmethod
    def from_dataframe(
        cls,
        activity_df: pd.DataFrame,
        descriptors_df: pd.DataFrame,
        target_column: str = "neg_log_value",
        use_structural_split: bool = True,
        similarity_cutoff: float = 0.7,
        holdout_size: float = 0.2,
        random_state: int = 42,
        ):
        """Creates a DatasetWrapper object from activity_df and descriptors_df obtained from other functions in the pipeline"""
        instance = cls()
        nonfeature_columns = activity_df.columns
        full_df = pd.concat([activity_df, descriptors_df], axis=1)
        print_low("Loading DatasetWrapper object from unsplit dataframes and splitting data.")
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
                similarity_cutoff=similarity_cutoff,
                holdout_size=holdout_size,
                random_state=random_state,
                )
        except Exception as e:
            print("Dataset loading failed.")
            raise e
        print_low(f"DatasetWrapper object loaded from unsplit DataFrames and split into train/test sets.")
        print_high(f"Dataset size: {instance.general_data.shape[0]}")
        print_high(f"Train subset size: {len(instance.y_train)}")
        print_high(f"Test subset size: {len(instance.y_test)}")
        print_high(f"Number of features: {instance.x_test.shape[1]}")
        return instance


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


    def describe(self) -> None:

        print(f"Dataset size: {self.general_data.shape[0]}")
        print(f"Train subset size: {len(self.y_train)}")
        print(f"Test subset size: {len(self.y_test)}")
        print(f"Number of features: {self.x_test.shape[1]}")


    def _load_from_path(self, file_path, target_col: str) -> None:
        """
        Loads the dataset from CSV files inside file_path folder.
        """
        self.general_data = pd.read_csv(f"{file_path}/general_data.csv", index_col="index")
        self.x_train = pd.read_csv(f"{file_path}/x_train.csv", index_col="index")
        self.x_test = pd.read_csv(f"{file_path}/x_test.csv", index_col="index")
        self.y_train = pd.read_csv(f"{file_path}/y_train.csv", index_col="index")[
            target_col
        ]
        self.y_test = pd.read_csv(f"{file_path}/y_test.csv", index_col="index")[
            target_col
        ]


    def _load_unsplit_dataframe(
        self,
        full_df: pd.DataFrame,
        target_column: str,
        nonfeature_columns,
        use_structural_split: bool,
        similarity_cutoff: float,
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
            train_index, test_index = scaffold_split(
                activity_df=self.general_data,
                test_size=holdout_size,
                similarity_cutoff=similarity_cutoff,
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


class PredictionData:

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
        deploy_descriptors: pd.DataFrame,
        model_features,
        ):
        print_low("Preparing DeployDatasetWrapper object.")
        instance = cls()
        instance.prepare_deploy_dataset(
            deploy_data=deploy_data,
            deploy_descriptors=deploy_descriptors,
            model_features=model_features
        )
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
        deploy_data,
        deploy_descriptors,
        model_features,
        ):
        self.deploy_data = deploy_data
        self.deploy_descriptors = deploy_descriptors
        print_high(f"Deployment descriptors shape before alignment: {self.deploy_descriptors.shape}")
        try:
            self.deploy_descriptors = self.deploy_descriptors.loc[:, model_features]
            print_high(f"Deployment descriptors shape after alignment: {self.deploy_descriptors.shape}")
        except KeyError as e:
            print("Failed to align with model features.")
            print_low(
                "Please, rerun prepare_deploy_dataset method from DeployDatasetWrapper instance with new model_features iterable.",
                )
            print_low(
                "Tip: use feature_names_in_ attribute from the model, or x_train.columns attribute from the training dataset wrapper.\n", )
            print_low(e)


def scaffold_split(
    activity_df: pd.DataFrame,
    smiles_col: str = 'canonical_smiles',
    test_size: float = 0.2,
    similarity_cutoff: float = 0.7,
    radius: int = 2,
    fingerprint_n_bits: int = 1024,
    ) -> tuple[pd.Index, pd.Index]:
    """
    Splits a DataFrame into train and test sets based on structural similarity
    using the Butina clustering algorithm.
    """
    print("\n--- Performing structural split ---")
    molecules: list = []
    for smiles in activity_df[smiles_col]:
        molecules.append(Chem.MolFromSmiles(smiles))
    fingerprint_generator = GetMorganGenerator(radius=radius, fpSize=fingerprint_n_bits)
    fingerprints: tuple = fingerprint_generator.GetFingerprints(molecules)
    distances = []
    n_mols = len(fingerprints)
    for i in range(n_mols):
        similarity_values = BulkTanimotoSimilarity(fingerprints[i], fingerprints[:i])
        distances.extend([1 - value for value in similarity_values])
    clusters: tuple[tuple] = Butina.ClusterData(distances, n_mols, 1.0 - similarity_cutoff, isDistData=True)
    clusters: list[tuple] = sorted(clusters, key=len, reverse=True)
    test_indices: list = []
    train_indices: list = []
    train_target = n_mols * (1 - test_size)
    test_target = n_mols * test_size

    for cluster in clusters:
        # Assign cluster to the set that is further from its target size
        train_need = len(train_indices) / train_target
        test_need = len(test_indices) / test_target

        if train_need >= test_need:
            test_indices.extend(cluster)
        else:
            train_indices.extend(cluster)

    train_df_indices = activity_df.index[train_indices]
    test_df_indices = activity_df.index[test_indices]
    print(f"Clustered {n_mols} molecules into {len(clusters)} clusters.")
    print(f"Train set size: {len(train_df_indices)}, Test set size: {len(test_df_indices)}.")
    print(f"Effective holdout ratio: {round(len(test_df_indices) / (len(test_df_indices) + len(train_df_indices)), 4)}")

    return train_df_indices, test_df_indices
