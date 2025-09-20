import numpy as np
import pandas as pd
from padelpy import padeldescriptor
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski


def calculate_fingerprint(dataframe, fingerprint):
    df_smi = dataframe['canonical_smiles']
    df_smi.to_csv('molecules.smi', sep='\t', index=False, header=False)
    print(
        '''\nBeginning descriptor calculation. This will create a descriptors.csv file.
        It can take a couple of hours or more, depending on your dataset size and descriptors chosen.
        You can check the progression at the descriptors.csv.log file that was created on this folder''',
        )
    padeldescriptor(
        mol_dir='molecules.smi',
        d_file='descriptors.csv',
        descriptortypes=fingerprint,
        detectaromaticity=True,
        standardizenitro=True,
        standardizetautomers=True,
        threads=-1,
        removesalt=True,
        log=True,
        fingerprints=True,
        )


def convert_to_m(molecules_df) -> pd.DataFrame:
    """método recebe um dataframe com dados em nM, uM, mM, ug.mL-1 e converte para M"""

    df_nm = molecules_df[molecules_df.standard_units.isin(['nM'])]
    df_um = molecules_df[molecules_df.standard_units.isin(['uM'])]
    df_mm = molecules_df[molecules_df.standard_units.isin(['mM'])]
    df_m = molecules_df[molecules_df.standard_units.isin(['M'])]
    df_ug_ml = pd.concat(
        [
            molecules_df[molecules_df.standard_units.isin(['ug.mL-1'])],
            molecules_df[molecules_df.standard_units.isin(['ug ml-1'])],
            ],
        )

    if not df_nm.empty and 'standard_value' in df_nm:
        df_nm.index = range(df_nm.shape[0])
        for i in df_nm.index:
            conc_nm = df_nm.iloc[i].standard_value
            conc_m = float(conc_nm) * 1e-9
            df_nm.standard_value.values[i] = conc_m
    else:
        pass

    if not df_um.empty and 'standard_value' in df_um:
        df_um.index = range(df_um.shape[0])
        for i in df_um.index:
            conc_um = df_um.iloc[i].standard_value
            conc_m = float(conc_um) * 1e-6
            df_um.standard_value.values[i] = conc_m
    else:
        pass

    if not df_mm.empty and 'standard_value' in df_mm:
        df_mm.index = range(df_mm.shape[0])
        for i in df_mm.index:
            conc_mm = df_mm.iloc[i].standard_value
            conc_m = float(conc_mm) * 1e-3
            df_mm.standard_value.values[i] = conc_m
    else:
        pass

    if not df_m.empty and 'standard_value' in df_m:
        df_m.loc['standard_value'] = df_m['standard_value'].astype(float)
    else:
        pass

    if not df_ug_ml.empty and 'standard_value' in df_ug_ml:
        df_ug_ml.index = range(df_ug_ml.shape[0])
        for i in df_ug_ml.index:
            conc_ug_ml = df_ug_ml.loc[i, 'standard_value']
            try:
                conc_g_l = float(conc_ug_ml) * 1e-3
            except ValueError as e:
                print(e, "standard_value not numeric, inserting nan")
                conc_g_l = np.nan
            conc_m = conc_g_l / df_ug_ml.loc[i, 'MW']
            df_ug_ml.standard_value.values[i] = conc_m

    dfs = [df_nm, df_um, df_mm, df_m, df_ug_ml]
    df_m = pd.concat(dfs, ignore_index=True)
    df_m.standard_units = 'M'
    return df_m


def get_lipinski_descriptors(molecules_df):
    molecules: list = []

    for elem in molecules_df['canonical_smiles']:
        mol = Chem.MolFromSmiles(elem)
        molecules.append(mol)

    base_data = np.arange(1, 1)
    i = 0

    for mol in molecules:
        desc_mol_wt = Descriptors.MolWt(mol)
        desc_mol_log_p = Descriptors.MolLogP(mol)
        desc_num_h_donors = Lipinski.NumHDonors(mol)
        desc_num_h_acceptors = Lipinski.NumHAcceptors(mol)
        row = np.array(
            [
                desc_mol_wt,
                desc_mol_log_p,
                desc_num_h_donors,
                desc_num_h_acceptors,
                ],
            )
        if i == 0:
            base_data = row
        else:
            base_data = np.vstack([base_data, row])
        i = i + 1

    column_names = ["MW", "LogP", "NumHDonors", "NumHAcceptors"]
    lipinski_descriptors = pd.DataFrame(data=base_data, columns=column_names, index=molecules_df.index)
    molecules_df_lipinski = pd.concat([molecules_df, lipinski_descriptors], axis=1)
    return molecules_df_lipinski


def get_ro5_violations(molecules_df):
    try:
        molecules_df["MW"]
    except KeyError as e:
        print(e, '\n', 'error: lipinski descriptors must be calculated before running this method')

    molecules_df_violations = molecules_df
    molecules_df_violations['Ro5Violations'] = 0

    for i in molecules_df.index:
        violations = 0
        if molecules_df_violations.at[i, 'MW'] >= 500:
            violations += 1
        if molecules_df_violations.at[i, 'LogP'] >= 5:
            violations += 1
        if molecules_df_violations.at[i, 'NumHDonors'] >= 5:
            violations += 1
        if molecules_df_violations.at[i, 'NumHAcceptors'] >= 10:
            violations += 1
        molecules_df_violations.at[i, 'Ro5Violations'] = violations

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


# """método recebe id chembl de uma molécula e retorna n de violações da rule of 5"""

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
