import os
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from numpy.random import seed
from scipy.stats import mannwhitneyu


def generate_unique_filename(base_path, base_name, tag_one='', tag_two='', suffix='.csv'):
    counter = 1
    if tag_one != '':
      tag_one = '_'+tag_one
    if tag_two != '':
      tag_two = '_'+tag_two
    while True:
        output_filename = f'{base_path}/{base_name}{tag_one}{tag_two}_{counter}{suffix}'
        if not os.path.exists(output_filename):
            return output_filename
        counter += 1


def check_if_int (input_string, default_output=0):
  """will check if input can be parsed as an integer. if so, will return the integer, and if not, will return default_output"""
  try:
    output = int(input_string)
    return output
  except TypeError as e:
    print(e,'\n','could not convert input to integer, returning default output = 0')
    output = default_output
    return output


def remove_low_variance_columns(input_data, threshold=0.1):
    selection = VarianceThreshold(threshold)
    selection.fit(input_data)
    return input_data[input_data.columns[selection.get_support(indices=True)]]


def normalize_value(molecules_df):
    norm = []
    molecules_df_norm = molecules_df

    for i in molecules_df_norm['standard_value']:
      if float(i) > 0.1:
        i = 0.1
      norm.append(i)

    molecules_df_norm['standard_value'] = norm
    return molecules_df_norm


def get_neg_log(molecules_df):
    neg_log = []
    molecules_df_neg_log = molecules_df

    for i in molecules_df_neg_log['standard_value']:
      i = float(i)
      neg_log.append(-np.log10(i))

    molecules_df_neg_log['neg_log_value'] = neg_log
    return molecules_df_neg_log


def mannwhitney_test(col_name:str, molecules_df1, molecules_df2, alpha:float=0.05):
  # Inspirado em: https://machinelearningmastery.com/nonparametric-statistical-significance-tests-in-python/
  seed(1)
  col1 = molecules_df1[col_name]
  col2 = molecules_df2[col_name]
  stat, p = mannwhitneyu(col1, col2)

  if p > alpha:
    interpretation = 'Same distribution (fail to reject H0)'
  else:
    interpretation = 'Different distributions (reject H0)'

  results = pd.DataFrame({'Descriptor':col_name,
                          'Statistics':stat,
                          'p':p,
                          'alpha':alpha,
                          'Interpretation':interpretation}, index=[0])
  # filename = 'mannwhitneyu_' + descriptor + '.csv'
  # results.to_csv(filename,index=False)
  return results


def treat_duplicates(molecules_df, method: str = 'median') -> pd.DataFrame:
  """
  Resolves duplicate molecule entries by applying an aggregation method to their
  'standard_value' and then dropping duplicates.

  Args:
      molecules_df (pd.DataFrame): DataFrame containing molecule data.
      method (str): The aggregation method to apply. One of ['median', 'mean',
                    'max', 'min']. Defaults to 'median'.

  Returns:
      pd.DataFrame: A DataFrame with duplicate molecules resolved.
  """
  print(f"Initial DataFrame size: {molecules_df.shape[0]}")
  treated_molecules_df = molecules_df.copy()
  # noinspection PyTypeChecker
  transformed_values = treated_molecules_df.groupby('molecule_chembl_id')['standard_value'].transform(method)
  treated_molecules_df.loc['standard_value'] = transformed_values
  treated_molecules_df = treated_molecules_df.drop_duplicates(subset='molecule_chembl_id')
  print(f"Filtered DataFrame size: {treated_molecules_df.shape[0]}")
  return treated_molecules_df
