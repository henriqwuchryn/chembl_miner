import os

import numpy as np
import pandas as pd
from padelpy import padeldescriptor
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski

from .utils import print_low, print_high
from .data_preprocessing import remove_low_variance_columns


def _calculate_fingerprint(
    activity_df: pd.DataFrame,
    fingerprint: str,
    smiles_col="canonical_smiles",
    ) -> pd.DataFrame:

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
    remove_low_variance: bool = False,
    low_variance_threshold: float = 0.0,
    ) -> pd.DataFrame:
    # TODO: generalizar para demais descritores
    print_low("Starting fingerprinters calculation.")
    print_high(
        "This will create temporary files in this folder: descriptors.csv; descriptors.csv.log and molecules.smi",
        )
    fingerprint_dict = {
        "atompairs2d"      : "fingerprinters/AtomPairs2DFingerprinter.xml",
        "atompairs2dcount" : "fingerprinters/AtomPairs2DFingerprintCount.xml",
        "estate"           : "fingerprinters/EStateFingerprinter.xml",
        "extended"         : "fingerprinters/ExtendedFingerprinter.xml",
        "fingerprinters"    : "fingerprinters/Fingerprinter.xml",
        "graphonly"        : "fingerprinters/GraphOnlyFingerprinter.xml",
        "klekota"          : "fingerprinters/KlekotaRothFingerprinter.xml",
        "klekotacount"     : "fingerprinters/KlekotaRothFingerprintCount.xml",
        "maccs"            : "fingerprinters/MACCSFingerprinter.xml",
        "pubchem"          : "fingerprinters/PubchemFingerprinter.xml",
        "substructure"     : "fingerprinters/SubstructureFingerprinter.xml",
        "substructurecount": "fingerprinters/SubstructureFingerprintCount.xml",
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
    if remove_low_variance:
        print_low("Removing low variance features.")
        print_high(f"Variance threshold: {low_variance_threshold}")
        initial_cols = descriptors_df.shape[1]
        descriptors_df = remove_low_variance_columns(
            input_data=descriptors_df,
            threshold=low_variance_threshold,
            )
        final_cols = descriptors_df.shape[1]
        print_high(f"Removed {initial_cols - final_cols} low variance columns. Kept {final_cols}.")
    print_low("Fingerprint calculation complete.")

    return descriptors_df


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
