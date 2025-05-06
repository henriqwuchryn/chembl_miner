import pandas as pd
import pickle
import os
import sys
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


    def subset_general_data(self, subset='train'):
        """
        Retrieves the corresponding rows from general_data for the train or test dataset.
        
        Args:
            subset (str): 'train' or 'test', indicating which subset to trace. Default value is 'train'.
        
        Returns:
            pd.DataFrame: Rows from general_data corresponding to the specified subset.
        """
        if subset == 'train':
            return self.general_data.loc[self.features_train.index]
        elif subset == 'test':
            return self.general_data.loc[self.features_test.index]
        else:
            raise ValueError("Subset must be 'train' or 'test'.")


    def save(self, file_path):
        """
        Saves the entire dataset to CSV files.
        """

        if not os.path.exists(file_path):
           os.mkdirs(file_path)

        self.general_data.to_csv(f'{file_path}/gd.csv', index_label='index')
        self.features_train.to_csv(f'{file_path}/ftr.csv', index_label='index')
        self.features_test.to_csv(f'{file_path}/fte.csv', index_label='index')
        self.target_train.to_csv(f'{file_path}/ttr.csv', index_label='index')
        self.target_test.to_csv(f'{file_path}/tte.csv', index_label='index')
        self.features_preprocessing.to_csv(f'{file_path}/fpr.csv', index_label='index')
        self.target_preprocessing.to_csv(f'{file_path}/tpr.csv', index_label='index')
        print(f"Dataset saved to {file_path}")

    def load(self, file_path):
        """
        Loads the dataset from CSV files.
        """
        try:
            self.general_data = pd.read_csv(f'{file_path}/gd.csv', index_col='index')
            self.features_train = pd.read_csv(f'{file_path}/ftr.csv', index_col='index')
            self.features_test = pd.read_csv(f'{file_path}/fte.csv', index_col='index')
            self.target_train = pd.read_csv(f'{file_path}/ttr.csv', index_col='index')
            self.target_test = pd.read_csv(f'{file_path}/tte.csv', index_col='index')
            self.features_preprocessing = pd.read_csv(f'{file_path}/fpr.csv', index_col='index')
            self.target_preprocessing = pd.read_csv(f'{file_path}/tpr.csv', index_col='index')
            print(f"Dataset loaded from {file_path}")
        except Exception as e:
            print(e)
            print("Dataset loading failed")

    def load_unsplit_csv(self, file_path, general_columns, target_column, test_size=0.2, random_state=42):
        """
        Loads an unsplit CSV file containing all columns and splits it into
        general_data, features_train/test, and target_train/test.
        
        Args:
            file_path (str): Path to the CSV file.
            general_columns (list): List of columns representing general data.
            target_column (str): Column name representing the target variable.
            test_size (float): Proportion of the dataset to include in the test split. Standard value = 0.2.
            random_state (int): Random state for reproducibility. Standard value = 42.
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
    def load_dataset(file_path):
        instance = DatasetWrapper()
        instance.load(file_path)
        return instance


    @staticmethod
    def load_raw_dataset(file_path, general_columns, target_column, test_size=0.2, random_state=42):
        instance = DatasetWrapper()
        instance.load_unsplit_csv(file_path, general_columns, target_column, test_size=0.2, random_state=42)
        return instance
