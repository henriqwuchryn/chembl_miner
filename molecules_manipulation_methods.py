import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
from padelpy import padeldescriptor, from_smiles


def calculate_fingerprint(dataframe, fingerprint):
    df_smi = dataframe['canonical_smiles']
    df_smi.to_csv('molecules.smi', sep='\t',  index=False, header=False)
    print('''\nBeginning descriptor calculation. This will create a descriptors.csv file.
It can take a couple of hours or more, depending on your dataset size and descriptors chosen.
You can check the progression at the descriptors.csv.log file that was created on this folder''')
    padeldescriptor(mol_dir='molecules.smi',
                    d_file='descriptors.csv',
                    descriptortypes=fingerprint,
                    detectaromaticity=True,
                    standardizenitro=True,
                    standardizetautomers=True,
                    threads=-1,
                    removesalt=True,
                    log=True,
                    fingerprints=True)


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