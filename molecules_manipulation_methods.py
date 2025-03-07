from chembl_webresource_client.new_client import new_client
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
from sklearn.feature_selection import VarianceThreshold


def generate_unique_filename(base_path, base_name, tag_one='', tag_two=''):
    counter = 1
    if tag_one != '':
      tag_one = '_'+tag_one
    if tag_two != '':
      tag_two = '_'+tag_two
    while True:
        output_filename = f'{base_path}/{base_name}{tag_one}{tag_two}{counter}.csv'
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


def convert_to_M (molecules_df) -> pd.DataFrame:
  """método recebe um dataframe com dados em nM, uM, mM, ug.mL-1 e converte para M"""

  df_nM = molecules_df[molecules_df.standard_units.isin(['nM'])]
  df_uM = molecules_df[molecules_df.standard_units.isin(['uM'])]
  df_mM = molecules_df[molecules_df.standard_units.isin(['mM'])]
  df_M = molecules_df[molecules_df.standard_units.isin(['M'])]
  df_ug_ml = pd.concat([molecules_df[molecules_df.standard_units.isin(['ug.mL-1'])],
                        molecules_df[molecules_df.standard_units.isin(['ug ml-1'])]])
  
  if not df_nM.empty and 'standard_value' in df_nM:
    df_nM.index = range(df_nM.shape[0])
    for i in df_nM.index:
      conc_nM = df_nM.iloc[i].standard_value
      conc_M = float(conc_nM) * 1e-9
      df_nM.standard_value.values[i] = conc_M
  else:
    pass

  if not df_uM.empty and 'standard_value' in df_uM:
    df_uM.index = range(df_uM.shape[0])
    for i in df_uM.index:
      conc_uM = df_uM.iloc[i].standard_value
      conc_M = float(conc_uM) * 1e-6
      df_uM.standard_value.values[i] = conc_M
  else:
    pass

  if not df_mM.empty and 'standard_value' in df_mM:
    df_mM.index = range(df_mM.shape[0])
    for i in df_mM.index:
      conc_mM = df_mM.iloc[i].standard_value
      conc_M = float(conc_mM) * 1e-3
      df_mM.standard_value.values[i] = conc_M
  else:
    pass

  if not df_M.empty and 'standard_value' in df_M:
    df_M.loc['standard_value'] = df_M['standard_value'].astype(float)
  else:
    pass

  if not df_ug_ml.empty and 'standard_value' in df_ug_ml:
    df_ug_ml.index = range(df_ug_ml.shape[0])
    for i in df_ug_ml.index:
      conc_ug_ml = df_ug_ml.iloc[i].standard_value
      try:
        conc_g_l = float(conc_ug_ml) * 1e-3
      except: 
        conc_g_l = np.nan
      conc_M = conc_g_l/df_ug_ml.loc[i,'MW']
      df_ug_ml.standard_value.values[i] = conc_M

  dfs = []
  dfs.append(df_nM)
  dfs.append(df_uM)
  dfs.append(df_mM)
  dfs.append(df_M)
  dfs.append(df_ug_ml)
  dfM = pd.concat(dfs, ignore_index=True)
  dfM.standard_units='M'
  return dfM


def getLipinskiDescriptors(molecules_df):
  molecules: list = []

  for elem in molecules_df['canonical_smiles']:
      mol = Chem.MolFromSmiles(elem)
      molecules.append(mol)

  baseData = np.arange(1,1)
  i=0

  for mol in molecules:
      desc_MolWt = Descriptors.MolWt(mol)
      desc_MolLogP = Descriptors.MolLogP(mol)
      desc_NumHDonors = Lipinski.NumHDonors(mol)
      desc_NumHAcceptors = Lipinski.NumHAcceptors(mol)
      row = np.array([desc_MolWt,
                      desc_MolLogP,
                      desc_NumHDonors,
                      desc_NumHAcceptors])
      if(i==0):
          baseData=row
      else:
          baseData=np.vstack([baseData, row])
      i=i+1

  columnNames=["MW","LogP","NumHDonors","NumHAcceptors"]
  lipinski_descriptors = pd.DataFrame(data=baseData,columns=columnNames)
  molecules_df_lipinski = pd.concat([molecules_df,lipinski_descriptors], axis=1)
  return molecules_df_lipinski


def getRo5Violations (molecules_df):
  try:
    'MW' in molecules_df.columns
  except:
    print('error: lipinski descriptors must be calculated before running this method')

  molecules_df_violations = molecules_df
  molecules_df_violations['Ro5Violations'] = 0

  for i in molecules_df.index:
    violations = 0
    if molecules_df_violations.at[i,'MW'] <= 500:
      violations += 1
    if molecules_df_violations.at[i,'LogP'] <= 5:
      violations += 1
    if molecules_df_violations.at[i,'NumHDonors'] <= 5:
      violations += 1
    if molecules_df_violations.at[i,'NumHAcceptors'] <= 10:
      violations += 1
    molecules_df_violations.at[i,'Ro5Violations'] = violations
  
  return molecules_df_violations


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
  from numpy.random import seed
  from scipy.stats import mannwhitneyu
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


# def getMolarMass (molecule_chembl_id) -> dict:
  
#   """método recebe id chembl de uma molecula e retorna massa molar"""

#   try:
#     mol_client == chembl_webresource_client.query_set.QuerySet
#   except:
#     mol_client = new_client.molecule

#   mol = mol_client.filter(chembl_id=molecule_chembl_id)
#   try:
#     massa_molar = float(mol[0]['molecule_properties']['full_mwt'])
#   except:
#     massa_molar = np.nan
#   return massa_molar



  # """método recebe id chembl de uma molécula e retorna n de violacoes da rule of 5"""

  # if mol_client != chembl_webresource_client.query_set.QuerySet:
  #   mol_client = new_client.molecule

  # mol = mol_client.filter(chembl_id=molecule_chembl_id)
  # try:
  #   ro5 = int(mol[0]['molecule_properties']['num_ro5_violations'])
  #   print(molecule_chembl_id," ok", end='\r')
  # except:
  #   ro5 = np.nan
  #   print(molecule_chembl_id," nan", end='\r')
  # return ro5