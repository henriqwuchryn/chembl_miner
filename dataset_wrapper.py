import pandas as pd
import pickle
import os
import sys
from sklearn.model_selection import train_test_split

class DatasetWrapper:
    def __init__(self, general_data=None, x_train=None, x_test=None, y_train=None, y_test=None):
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
        self.file_path = ''


    def subset_general_data(self, subset='train'):
        """
        Retrieves the corresponding rows from general_data for the train or test dataset.
        
        Args:
            subset (str): 'train' or 'test', indicating which subset to trace. Default value is 'train'.
        
        Returns:
            pd.DataFrame: Rows from general_data corresponding to the specified subset.
        """
        if subset == 'train':
            return self.general_data.loc[self.x_train.index]
        elif subset == 'test':
            return self.general_data.loc[self.x_test.index]
        else:
            raise ValueError("Subset must be 'train' or 'test'.")


    def save(self, file_path):
        """
        Saves the entire dataset to CSV files.
        """

        if not os.path.exists(file_path):
           os.makedirs(file_path)

        self.general_data.to_csv(f'{file_path}/gd.csv', index_label='index')
        self.x_train.to_csv(f'{file_path}/ftr.csv', index_label='index')
        self.x_test.to_csv(f'{file_path}/fte.csv', index_label='index')
        self.y_train.to_csv(f'{file_path}/ttr.csv', index_label='index')
        self.y_test.to_csv(f'{file_path}/tte.csv', index_label='index')
        self.x_preprocessing.to_csv(f'{file_path}/fpr.csv', index_label='index')
        self.y_preprocessing.to_csv(f'{file_path}/tpr.csv', index_label='index')
        print(f"Dataset saved to {file_path}")

    def load(self, file_path):
        """
        Loads the dataset from CSV files.
        """
        try:
            self.general_data = pd.read_csv(f'{file_path}/gd.csv', index_col='index')
            self.x_train = pd.read_csv(f'{file_path}/ftr.csv', index_col='index')
            self.x_test = pd.read_csv(f'{file_path}/fte.csv', index_col='index')
            self.y_train = pd.read_csv(f'{file_path}/ttr.csv', index_col='index')['neg_log_value']
            self.y_test = pd.read_csv(f'{file_path}/tte.csv', index_col='index')['neg_log_value']
            self.x_preprocessing = pd.read_csv(f'{file_path}/fpr.csv', index_col='index')
            self.y_preprocessing = pd.read_csv(f'{file_path}/tpr.csv', index_col='index')['neg_log_value']
            self.file_path = file_path
            print(f"Dataset loaded from {file_path}")
        except Exception as e:
            print(e)
            print("Dataset loading failed")

    def load_unsplit_csv(self, file_path, general_columns, target_column, test_size=0.2, random_state=42):
        """
        Loads an unsplit CSV file containing all columns and splits it into
        general_data, x_train/test, and y_train/test.
        
        Args:
            file_path (str): Path to the CSV file.
            general_columns (list): List of columns representing general data.
            target_column (str): Column name representing the target variable.
            test_size (float): Proportion of the dataset to include in the test split. Standard value = 0.2.
            random_state (int): Random state for reproducibility. Standard value = 42.
        """
        try:
            full_df = pd.read_csv(file_path, index_col='index')
        except Exception as e:
            print(e)
            print('\nUnsplit CSV file does not exist')
        try:
            self.general_data = full_df[general_columns]
            target = full_df[target_column]
            not_feature_columns = general_columns + [target_column]
            features = full_df.drop(columns=not_feature_columns)
        except KeyError as e:
            print(e)
            print('\nUnsplit CSV file does not contain the correct columns')
        # Split the dataset
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            features, target, test_size=test_size, random_state=random_state
        )
        self.file_path = file_path
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
        print(f'\nDataset obtained from {self.file_path}\nDataset size: {size}\nNumber of features: {features}\nTrain subset size: {size_train}\nTest subset size: {size_test}')


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
