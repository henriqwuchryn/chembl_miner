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


def check_if_int (input, default_output=0):
  """will check if input can be parsed as an integer. if so, will return the integer, and if not, will return defaul_output"""
  try:
    output = int(input)
    return output
  except:
    output = default_output
    return output


def remove_low_variance_columns(input_data, threshold=0.1):
    selection = VarianceThreshold(threshold)
    selection.fit(input_data)
    return input_data[input_data.columns[selection.get_support(indices=True)]]


def normalizeValue(molecules_df):
    norm = []
    molecules_df_norm = molecules_df

    for i in molecules_df_norm['standard_value']:
      if float(i) > 0.1:
        i = 0.1
      norm.append(i)

    molecules_df_norm['standard_value'] = norm
    return molecules_df_norm


def getNegLog(molecules_df):
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
