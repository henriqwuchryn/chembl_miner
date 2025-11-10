import os

import numpy as np
import pandas as pd
from padelpy import padeldescriptor
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski

from .utils import print_low, print_high


def _calculate_fingerprint(
    activity_df: pd.DataFrame,
    fingerprint: str,
    smiles_col="canonical_smiles",
    ) -> pd.DataFrame:
    """
    Internal helper function to run PaDEL for a single descriptor XML.

    Creates temporary files ('molecules.smi', 'descriptors.csv') and
    cleans them up after execution.

    Args:
        activity_df (pd.DataFrame): DataFrame with a SMILES column.
        fingerprint (str): The file path to the PaDEL descriptor XML file.
        smiles_col (str, optional): The name of the SMILES column.
            Defaults to "canonical_smiles".

    Returns:
        pd.DataFrame: A DataFrame of the calculated descriptors, indexed
            identically to `activity_df`.
    
    Example:
        ```python
        # This is an internal function, typically called by 
        # calculate_fingerprint().
        
        # You would need a valid PaDEL XML file path.
        xml_path = "path/to/PubchemFingerprinter.xml" 
        try:
            _calculate_fingerprint(activity_df, fingerprint=xml_path)
        except Exception as e:
            print(f"PaDEL execution failed: {e}")
        ```
    """

    df_smi = activity_df[smiles_col]
    df_smi.to_csv('molecules.smi', sep='\t', index=False, header=False)
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
    descriptors_df_i = pd.read_csv("descriptors.csv")
    descriptors_df_i = descriptors_df_i.drop("Name", axis=1)
    descriptors_df_i = pd.DataFrame(data=descriptors_df_i, index=activity_df.index)
    os.remove("descriptors.csv")
    os.remove("descriptors.csv.log")
    os.remove("molecules.smi")
    return descriptors_df_i


def calculate_fingerprint(
    activity_df: pd.DataFrame,
    smiles_col="canonical_smiles",
    fingerprint: str | list[str] = "pubchem",
    ) -> pd.DataFrame:
    """
    Calculates one or more molecular fingerprints using PaDEL-Descriptor.

    This function uses predefined XML configuration files for various
    fingerprint types (e.g., "pubchem", "maccs", "estate").

    Args:
        activity_df (pd.DataFrame): DataFrame containing a SMILES column.
        smiles_col (str, optional): The name of the SMILES column.
            Defaults to "canonical_smiles".
        fingerprint (str | list[str], optional): The name(s) of the
            fingerprint(s) to calculate. Available options include "pubchem",
            "maccs", "estate", "klekota", etc. Defaults to "pubchem".

    Returns:
        pd.DataFrame: A DataFrame containing the calculated descriptors
            (fingerprints) for all molecules.
    
    Example:
        ```python
        # Create mock data
        data = {'canonical_smiles': ['CCO', 'CCC', 'c1ccccc1']}
        activity_df = pd.DataFrame(data)
        
        # Calculate a single fingerprint (PubChem)
        try:
            pubchem_fp = calculate_fingerprint(
                activity_df, 
                fingerprint="pubchem"
            )
            print(pubchem_fp.shape) 
            # (3, 881)
        
            # Calculate multiple fingerprints
            multi_fp = calculate_fingerprint(
                activity_df, 
                fingerprint=['maccs', 'estate']
            )
            print(multi_fp.shape)
            # (3, 244) -> 166 (MACCS) + 78 (EState)
        
        except Exception as e:
            print(f"PaDEL calculation failed. Is PaDEL installed? {e}")
        ```
    """
    # TODO: generalizar para demais descritores
    print_low("Starting fingerprinters calculation.")
    print_high(
        "This will create temporary files in this folder: descriptors.csv; descriptors.csv.log and molecules.smi",
        )
    script_dir = os.path.dirname(os.path.abspath(__file__))
    fingerprinters_dir = os.path.join(script_dir, 'fingerprinters')
    fingerprint_dict = {
        "atompairs2d"      : os.path.join(fingerprinters_dir, "AtomPairs2DFingerprinter.xml"),
        "atompairs2dcount" : os.path.join(fingerprinters_dir, "AtomPairs2DFingerprintCount.xml"),
        "estate"           : os.path.join(fingerprinters_dir, "EStateFingerprinter.xml"),
        "extended"         : os.path.join(fingerprinters_dir, "ExtendedFingerprinter.xml"),
        "fingerprinters"    : os.path.join(fingerprinters_dir, "Fingerprinter.xml"),
        "graphonly"        : os.path.join(fingerprinters_dir, "GraphOnlyFingerprinter.xml"),
        "klekota"          : os.path.join(fingerprinters_dir, "KlekotaRothFingerprinter.xml"),
        "klekotacount"     : os.path.join(fingerprinters_dir, "KlekotaRothFingerprintCount.xml"),
        "maccs"            : os.path.join(fingerprinters_dir, "MACCSFingerprinter.xml"),
        "pubchem"          : os.path.join(fingerprinters_dir, "PubchemFingerprinter.xml"),
        "substructure"     : os.path.join(fingerprinters_dir, "SubstructureFingerprinter.xml"),
        "substructurecount": os.path.join(fingerprinters_dir, "SubstructureFingerprintCount.xml"),
    }

    if type(fingerprint) == str:
        fingerprint = [fingerprint]

    descriptors_df = pd.DataFrame(index=activity_df.index)

    for i in fingerprint:
        print_high(f"Calculating '{i}' fingerprinters.")
        fingerprint_path = fingerprint_dict[i]
        descriptors_df_i = _calculate_fingerprint(
            activity_df=activity_df,
            smiles_col=smiles_col,
            fingerprint=fingerprint_path,
            )
        descriptors_df = pd.concat(objs=[descriptors_df, descriptors_df_i], axis=1)
    print_high(f"Total features from fingerprints: {descriptors_df.shape[1]}")
    print_low("Fingerprint calculation complete.")

    return descriptors_df


def get_lipinski_descriptors(molecules_df):
    """
    Calculates Lipinski descriptors using RDKit.

    Calculates Molecular Weight (MW), LogP, Number of H-Bond Donors,
    and Number of H-Bond Acceptors.

    Args:
        molecules_df (pd.DataFrame): DataFrame with a 'canonical_smiles' column.

    Returns:
        pd.DataFrame: The original DataFrame concatenated with the new
            Lipinski descriptor columns ("MW", "LogP", "NumHDonors",
            "NumHAcceptors").
    
    Example:
        ```python
        data = {
            'canonical_smiles': ['CCO', 'c1ccccc1C(=O)O', 'invalid_smiles'],
            'other_data': [1, 2, 3]
        }
        df = pd.DataFrame(data)
        
        df_with_lipinski = get_lipinski_descriptors(df)
        
        print(df_with_lipinski)
        #   canonical_smiles  other_data          MW      LogP  NumHDonors  NumHAcceptors
        # 0              CCO           1   46.069000  0.295200         1.0            1.0
        # 1   c1ccccc1C(=O)O           2  122.123000  1.849000         1.0            2.0
        # 2   invalid_smiles           3         NaN       NaN         NaN            NaN
        ```
    """

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
