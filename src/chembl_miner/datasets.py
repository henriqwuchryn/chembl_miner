import os

import pandas as pd
from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.DataStructs import BulkTanimotoSimilarity
from rdkit.ML.Cluster import Butina
from sklearn.model_selection import train_test_split

from .utils import print_low, print_high


class TrainingData:
    """
    A data wrapper class to store, manage, and split QSAR training data.

    This class holds the general (non-feature) data, as well as the
    feature matrices (x) and target vectors (y) for both training and
    testing sets.
    """

    def __init__(
        self,
        general_data=pd.DataFrame(),
        x_train=pd.DataFrame(),
        x_test=pd.DataFrame(),
        y_train=pd.Series(),
        y_test=pd.Series(),
        ):
        """
        Initializes the TrainingData object, but using from_dataframe or from_path is advised.
        """
        self.general_data = general_data
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test


    @classmethod
    def from_path(cls, file_path, target_col: str = "neg_log_value"):
        """
        Loads a TrainingData object from a directory of saved CSV files.

        Assumes the directory was created with the to_path function and contains
        'general_data.csv', 'x_train.csv', 'x_test.csv', 'y_train.csv', and
        'y_test.csv'.

        Args:
            file_path (str): The path to the directory containing the data files.
            target_col (str, optional): The name of the target column in the
                y_train/y_test CSV files. Defaults to "neg_log_value".

        Returns:
            TrainingData: An instance of TrainingData populated with the
                loaded data.
        
        Example:
            ```python
            # Assume "my_saved_dataset" is a folder created by .to_path()
            try:
                dataset = TrainingData.from_path(
                    "my_saved_dataset",
                    target_col="neg_log_value"
                )
                dataset.describe()
            except FileNotFoundError:
                print("Directory 'my_saved_dataset' not found.")
            ```
        """
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
        """
        Creates a TrainingData object from unsplit activity and descriptor DataFrames.

        This method combines the DataFrames, splits them into train/test sets
        (using either random or structural splitting), and populates the
        TrainingData instance.

        Args:
            activity_df (pd.DataFrame): DataFrame containing non-feature data
                (e.g., SMILES) and the target column.
            descriptors_df (pd.DataFrame): DataFrame containing the feature
                (descriptor) columns. Must share the same index as activity_df.
            target_column (str, optional): The name of the target variable
                in `activity_df`. Defaults to "neg_log_value".
            use_structural_split (bool, optional): If True, uses a scaffold
                split (Butina clustering). If False, uses a random split.
                Defaults to True.
            similarity_cutoff (float, optional): Tanimoto similarity cutoff
                for structural splitting. Defaults to 0.7.
            holdout_size (float, optional): The fraction of data to allocate
                to the test set. Defaults to 0.2.
            random_state (int, optional): The random state for reproducibility
                (used only if `use_structural_split` is False). Defaults to 42.

        Returns:
            TrainingData: An instance of TrainingData populated with the
                split data.
        
        Example:
            ```python
            # Create mock data
            activity_data = {
                'canonical_smiles': ['CCC', 'CCO', 'CCN', 'CCCC', 'CCF', 'CCCl'],
                'neg_log_value': [6.1, 6.2, 6.3, 7.1, 7.2, 7.3]
            }
            activity_df = pd.DataFrame(activity_data)
            
            descriptor_data = {
                'feat1': [1, 2, 1, 3, 2, 3],
                'feat2': [5, 6, 5, 7, 6, 7]
            }
            descriptors_df = pd.DataFrame(descriptor_data)
            
            # Create dataset using structural split (default)
            dataset = TrainingData.from_dataframe(
                activity_df,
                descriptors_df,
                target_column="neg_log_value",
                holdout_size=0.33
            )
            
            # print(dataset.x_train.shape, dataset.x_test.shape)
            # (4, 2) (2, 2)
            ```
        """
        instance = cls()
        nonfeature_columns = activity_df.columns
        if descriptors_df.isna().any().any():
            descriptors_df = descriptors_df.dropna(how="any")
            n_rows_dropped = activity_df.shape[0] - descriptors_df.shape[0]
            activity_df = activity_df.loc[descriptors_df.index]
            print_low(f'There was a NA in descriptors DataFrame, {n_rows_dropped} rows dropped')
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
        Saves all components of the dataset to CSV files in a specified directory.

        Files created: 'general_data.csv', 'x_train.csv', 'x_test.csv',
        'y_train.csv', 'y_test.csv'.

        Created files can be parsed using from_path function.

        Args:
            file_path (str): The path to the directory where files will be saved.
                The directory will be created if it doesn't exist.
        
        Example:
            ```python
            # Assuming 'dataset' is a populated TrainingData object
            save_directory = "my_saved_dataset"
            dataset.to_path(save_directory)
            
            # This creates files like:
            # - my_saved_dataset/general_data.csv
            # - my_saved_dataset/x_train.csv
            # - ...etc.
            ```
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
        Retrieves the general_data rows corresponding to the train or test set.

        Args:
            train_subset (bool, optional): If True, returns data for the
                training set. If False, returns data for the test set.
                Defaults to True.

        Returns:
            pd.DataFrame: The subset of `general_data` corresponding to
                the selected (train/test) indices.
        
        Example:
            ```python
            # Assuming 'dataset' is a populated TrainingData object
            
            # Get general info (like SMILES) for the training set
            train_info_df = dataset.subset_general_data(train_subset=True)
            
            # Get general info (like SMILES) for the test set
            test_info_df = dataset.subset_general_data(train_subset=False)
            
            # print(train_info_df.shape, test_info_df.shape)
            ```
        """
        subset_type = "train" if train_subset else "test"
        print_high(f"Subsetting general_data for the {subset_type} set.")
        if train_subset:
            return self.general_data.loc[self.x_train.index]
        else:
            return self.general_data.loc[self.x_test.index]


    def describe(self) -> None:
        """
        Prints a summary of the dataset's dimensions.

        Includes total size, train size, test size, and number of features.
        
        Example:
            ```python
            # Assuming 'dataset' is a populated TrainingData object
            dataset.describe()
            
            # Example Output:
            # Dataset size: 1000
            # Train subset size: 800
            # Test subset size: 200
            # Number of features: 881
            ```
        """

        print(f"Dataset size: {self.general_data.shape[0]}")
        print(f"Train subset size: {len(self.y_train)}")
        print(f"Test subset size: {len(self.y_test)}")
        print(f"Number of features: {self.x_test.shape[1]}")


    def _load_from_path(self, file_path, target_col: str) -> None:
        """
        Internal helper method to load data from CSV files.

        Args:
            file_path (str): Path to the directory.
            target_col (str): The name of the target column in y_train/y_test.
        
        Example:
            ```python
            # This is an internal method, typically called by from_path()
            instance = TrainingData()
            instance._load_from_path("my_saved_dataset", "neg_log_value")
            ```
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
        Internal helper method to split a full DataFrame and populate the object.

        Args:
            full_df (pd.DataFrame): The combined DataFrame.
            target_column (str): Name of the target column.
            nonfeature_columns (pd.Index): Columns to be stored in `general_data`.
            use_structural_split (bool): Whether to use scaffold split.
            similarity_cutoff (float): Cutoff for scaffold split.
            holdout_size (float): Fraction for the test set.
            random_state (int): Random state for random split.
        
        Example:
            ```python
            # This is an internal method, typically called by from_dataframe()
            # ... (see from_dataframe() for example data) ...
            instance = TrainingData()
            instance._load_unsplit_dataframe(
                full_df, "neg_log_value", activity_df.columns, 
                True, 0.7, 0.2, 42
            )
            ```
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
    """
    A data wrapper class to store and manage data for deployment/prediction.

    This class holds the general data for new molecules, their calculated
    descriptors, and a dictionary to store multiple model predictions.
    """

    def __init__(
        self,
        deploy_data: pd.DataFrame = None,
        deploy_descriptors: pd.DataFrame = None,
        prediction: dict = None,
        ) -> None:
        """
        Initializes the PredictionData object.

        Args:
            deploy_data (pd.DataFrame, optional): DataFrame containing
                non-feature data (e.g., SMILES) for new molecules.
                Defaults to None.
            deploy_descriptors (pd.DataFrame, optional): DataFrame of
                descriptors for the new molecules. Defaults to None.
            prediction (dict, optional): A dictionary to store prediction
                results. Defaults to None.
        
        Example:
            ```python
            # Create mock data for new molecules
            new_mols_df = pd.DataFrame({'smiles': ['C(C)C', 'C(C)N']})
            new_descriptors_df = pd.DataFrame({'feat1': [1.5, 2.5], 'feat2': [3.5, 4.5]})
            
            pred_data = PredictionData(
                deploy_data=new_mols_df,
                deploy_descriptors=new_descriptors_df
            )
            ```
        """

        self.deploy_data = pd.DataFrame() if deploy_data is None else deploy_data
        self.deploy_descriptors = pd.DataFrame() if deploy_descriptors is None else deploy_descriptors
        self.prediction = {} if prediction is None else prediction


    @classmethod
    def prepare_dataset(
        cls,
        deploy_data: pd.DataFrame,
        deploy_descriptors: pd.DataFrame,
        model_features,
        ):
        """
        Creates and prepares a PredictionData object.

        This method aligns the columns of `deploy_descriptors` with the
        features used by a trained model.

        Args:
            deploy_data (pd.DataFrame): DataFrame with non-feature data.
            deploy_descriptors (pd.DataFrame): DataFrame with descriptors.
            model_features (list or pd.Index): A list of feature names
                that the model expects (e.g., from `model.feature_names_in_`).

        Returns:
            PredictionData: A new, prepared instance of PredictionData.
        
        Example:
            ```python
            # Mock data for new molecules
            new_mols_df = pd.DataFrame({'smiles': ['C(C)C', 'C(C)N']})
            
            # Descriptors might have extra columns
            new_descriptors_df = pd.DataFrame({
                'feat1': [1.5, 2.5], 
                'feat2': [3.5, 4.5],
                'extra_feat': [0, 0]
            })
            
            # Features the model was trained on can be obtained by acessing
            # the .feature_names_in attribute on a scikit-learn estimator
            model_cols = model.feature_names_in_

            print(model_cols)
            # ['feat1', 'feat2']
            
            pred_data = PredictionData.prepare_dataset(
                new_mols_df,
                new_descriptors_df,
                model_features=model_cols
            )
            
            # print(pred_data.deploy_descriptors.columns)
            # Index(['feat1', 'feat2'], dtype='object')
            ```
        """

        print_low("Preparing DeployDatasetWrapper object.")
        instance = cls()
        instance.prepare_deploy_dataset(
            deploy_data=deploy_data,
            deploy_descriptors=deploy_descriptors,
            model_features=model_features
        )
        print_low("DeployDatasetWrapper object prepared.")
        return instance

    #TODO:  IMPLEMENT BOTH DILL AND HUMAN READABLE SAVE MODES (CSV AND TXT FILES READABLE WITH A TEXT EDITOR)
    @classmethod
    def from_path(cls, file_path):
        """
        Loads a PredictionData object from a directory of saved CSV files.

        Loads 'deploy_data.csv', 'deploy_descriptors.csv', and any CSV files
        found in the '/prediction/' subdirectory.

        Args:
            file_path (str): The path to the directory containing the data.

        Returns:
            PredictionData: An instance populated with the loaded data.
        
        Example:
            ```python
            # Assume "my_prediction_run" is a folder created by .to_path()
            try:
                pred_data = PredictionData.from_path("my_prediction_run")
                
                print(pred_data.deploy_data.head())
                print(pred_data.prediction.keys())
                # dict_keys(['xgboost_reg_1'])
            
            except FileNotFoundError:
                print("Directory 'my_prediction_run' not found.")
            ```
        """
        if not os.path.exists(file_path):
            print("Provided file_path does not exist")
        print_low(f"Loading DeployDatasetWrapper object from {file_path}.")
        instance = cls()
        instance.deploy_data = pd.read_csv(f"{file_path}/deploy_data.csv", index_col='index')
        instance.deploy_descriptors = pd.read_csv(f"{file_path}/deploy_descriptors.csv",index_col='index')
        predictions_path = f"{file_path}/prediction/"
        predictions_files = os.listdir(predictions_path)
        for file in predictions_files:
            key = file[:-4]
            instance.prediction[key] = pd.read_csv(f"{predictions_path}/{file}", index_col='index')
        print_low("DeploymentDatasetWrapper object with data, descriptors, and previous predictions loaded.")
        return instance


    def to_path(self, file_path):
        """
        Saves all components of the dataset to CSV files in a specified directory.

        Saves 'deploy_data.csv', 'deploy_descriptors.csv', and saves each
        prediction in the `prediction` dictionary to a separate CSV in a
        '/prediction/' subdirectory.

        Args:
            file_path (str): The path to the directory where files will be saved.
                The directory will be created if it doesn't exist.
        
        Example:
            ```python
            # Assuming 'pred_data' is a populated PredictionData object
            # and it has a prediction stored
            # pred_data.prediction['xgboost_reg_1'] = pd.Series([8.1, 7.5])
            
            save_directory = "my_prediction_run"
            pred_data.to_path(save_directory)
            
            # This creates:
            # - my_prediction_run/deploy_data.csv
            # - my_prediction_run/deploy_descriptors.csv
            # - my_prediction_run/prediction/xgboost_reg_1.csv
            ```
        """

        print_low(f"Saving DeploymentDatasetWrapper object to {file_path}.")
        if not os.path.exists(file_path):
            print_high(f"Creating directory: {file_path}")
            os.makedirs(file_path)

        self.deploy_data.to_csv(f"{file_path}/deploy_data.csv", index_label="index")
        self.deploy_descriptors.to_csv(f"{file_path}/deploy_descriptors.csv", index_label="index")
        predictions_path = f"{file_path}/prediction/"
        if not os.path.exists(predictions_path):
            print_high(f"Creating directory: {predictions_path}")
            os.makedirs(predictions_path)
        for key in self.prediction.keys():
            self.prediction[key].to_csv(f"{predictions_path}/{key}.csv")
        print_low("DeploymentDatasetWrapper object with data, descriptors, and predictions saved.")


    def prepare_deploy_dataset(
        self,
        deploy_data,
        deploy_descriptors,
        model_features,
        ):
        """
        Prepares deployment data by dropping NAs and aligning descriptor columns.

        Ensures that the `deploy_descriptors` DataFrame has the exact same
        columns in the same order as the features used to train the model.

        This is an internal method, typically called by prepare_dataset()

        Args:
            deploy_data (pd.DataFrame): DataFrame with non-feature data.
            deploy_descriptors (pd.DataFrame): DataFrame with descriptors.
            model_features (list or pd.Index): A list of feature names
                that the model expects (e.g., from `model.feature_names_in_`).
        
        Example:
            ```python
            
            pred_data = PredictionData()
            # ... (see prepare_dataset() for example data) ...
            pred_data.prepare_deploy_dataset(
                new_mols_df, new_descriptors_df, model_cols
            )
            print(pred_data.deploy_descriptors.columns)
            ```
        """

        if deploy_descriptors.isna().any().any():
            deploy_descriptors = deploy_descriptors.dropna(how="any")
            n_rows_dropped = deploy_data.shape[0] - deploy_descriptors.shape[0]
            deploy_data = deploy_data.loc[deploy_descriptors.index]
            print_low(f'There was a NA in descriptors DataFrame, {n_rows_dropped} rows dropped')
        print_high(f"Deployment descriptors shape before alignment: {deploy_descriptors.shape}")
        try:
            deploy_descriptors = deploy_descriptors.loc[:, model_features]
            print_high(f"Deployment descriptors shape after alignment: {deploy_descriptors.shape}")
            self.deploy_data = deploy_data
            self.deploy_descriptors = deploy_descriptors
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
    Splits a DataFrame into train/test sets based on structural similarity.

    Uses Morgan fingerprints and Butina clustering to group structurally
    similar molecules, then allocates entire clusters to either the
    train or test set to create a more challenging and realistic split.

    This method is automatically called on TrainingData.from_dataframe() function.

    Args:
        activity_df (pd.DataFrame): DataFrame containing at least the
            SMILES column.
        smiles_col (str, optional): The name of the SMILES column.
            Defaults to 'canonical_smiles'.
        test_size (float, optional): The desired fraction of the test set.
            The actual split may differ slightly. Defaults to 0.2.
        similarity_cutoff (float, optional): The Tanimoto similarity
            threshold for clustering. Defaults to 0.7.
        radius (int, optional): The radius for the Morgan fingerprint.
            Defaults to 2.
        fingerprint_n_bits (int, optional): The number of bits for the
            Morgan fingerprint. Defaults to 1024.

    Returns:
        tuple[pd.Index, pd.Index]: A tuple containing the
            DataFrame indices for the training set and the test set.
    
    Example:
        ```python
        # Create mock data with two distinct clusters
        smiles_data = {
            'canonical_smiles': [
                'C1CCCCC1', 'C1CCCCC1C', 'C1CCCCC1CC', # Cluster 1
                'c1ccccc1', 'c1ccccc1C', 'c1ccccc1CC'  # Cluster 2
            ],
            'neg_log_value': [5.0, 5.1, 5.2, 8.0, 8.1, 8.2]
        }
        activity_df = pd.DataFrame(smiles_data)
        
        # Split with 33% test size
        # This will likely put one cluster in train and one in test
        train_idx, test_idx = scaffold_split(
            activity_df,
            test_size=0.33,
            similarity_cutoff=0.4 
        )
        
        print(f"Train indices: {train_idx.tolist()}")
        print(f"Test indices: {test_idx.tolist()}")
        
        # Example Output (indices may vary):
        # Train indices: [3, 4, 5]
        # Test indices: [0, 1, 2]
        ```
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
