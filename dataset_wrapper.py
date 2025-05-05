import pandas as pd
from sklearn.model_selection import train_test_split

class DatasetWrapper:
    def __init__(self, general_data=None, features_train=None, features_test=None, target_train=None, target_test=None):
        """
        Initializes the DatasetWrapper with optional dataframes.
        """
        self.general_data = general_data if general_data is not None else pd.DataFrame()
        self.features_train = features_train if features_train is not None else pd.DataFrame()
        self.features_test = features_test if features_test is not None else pd.DataFrame()
        self.target_train = target_train if target_train is not None else pd.DataFrame()
        self.target_test = target_test if target_test is not None else pd.DataFrame()
        self.features_preprocessing = pd.DataFrame()
        self.target_preprocessing = pd.DataFrame()

    def ensure_fixed_indices(self):
        """
        Ensures the indices are fixed across all dataframes for traceability.
        """
        if not self.general_data.empty:
            base_index = self.general_data.index
            self.features_train = self.features_train.reindex(base_index)
            self.features_test = self.features_test.reindex(base_index)
            self.target_train = self.target_train.reindex(base_index)
            self.target_test = self.target_test.reindex(base_index)

    def save_to_csv(self, file_path):
        """
        Saves the entire dataset to a single CSV file.
        """
        combined = {
            "general_data": self.general_data,
            "features_train": self.features_train,
            "features_test": self.features_test,
            "target_train": self.target_train,
            "target_test": self.target_test,
            "features_preprocessing": self.features_preprocessing,
            "target_preprocessing": self.target_preprocessing,
        }
        # Add keys to differentiate dataframes when saving
        output_df = pd.concat(combined, axis=1)
        output_df.to_csv(file_path)
        print(f"Dataset saved to {file_path}")

    def load_from_csv(self, file_path):
        """
        Loads the dataset from a single CSV file.
        """
        combined_df = pd.read_csv(file_path, header=[0, 1], index_col=0)
        self.general_data = combined_df["general_data"]
        self.features_train = combined_df["features_train"]
        self.features_test = combined_df["features_test"]
        self.target_train = combined_df["target_train"]
        self.target_test = combined_df["target_test"]
        self.features_preprocessing = combined_df.get("features_preprocessing", pd.DataFrame())
        self.target_preprocessing = combined_df.get("target_preprocessing", pd.DataFrame())
        print(f"Dataset loaded from {file_path}")

    def load_unsplit_csv(self, file_path, general_columns, target_column, test_size=0.2, random_state=42):
        """
        Loads an unsplit CSV file containing all columns and splits it into
        general_data, features_train/test, and target_train/test.
        
        Args:
            file_path (str): Path to the CSV file.
            general_columns (list): List of columns representing general data.
            target_column (str): Column name representing the target variable.
            test_size (float): Proportion of the dataset to include in the test split.
            random_state (int): Random state for reproducibility.
        """
        # Load the unsplit CSV file
        full_df = pd.read_csv(file_path, index_col=0)
        
        # Extract general data
        self.general_data = full_df[general_columns]

        # Exclude general and target columns to get feature columns dynamically
        features = full_df.drop(columns=general_columns + [target_column])
        target = full_df[target_column]

        # Split the dataset
        self.features_train, self.features_test, self.target_train, self.target_test = train_test_split(
            features, target, test_size=test_size, random_state=random_state
        )
        print(f"Unsplit dataset loaded and split into train/test sets from {file_path}")

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
        if len(self.features_train) > 10000:
            self.features_preprocessing = self.features_train.sample(n=max_samples, random_state=42)
            self.target_preprocessing = self.target_train.loc[self.features_preprocessing.index]
            print(f"Preprocessing dataset created with {max_samples} samples.")
        else:
            self.features_preprocessing = self.features_train
            self.target_preprocessing = self.target_train
            print("Preprocessing dataset uses the entire train dataset.")

    @staticmethod
    def from_csv(file_path):
        """
        Static method to create an instance of DatasetWrapper from a CSV file.
        """
        instance = DatasetWrapper()
        instance.load_from_csv(file_path)
        return instance