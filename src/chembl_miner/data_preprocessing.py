import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold

from .feature_engineering import get_lipinski_descriptors
from .data_retrieval import filter_by_assay
from .utils import print_low, print_high


def preprocess_data(
    activity_df: pd.DataFrame,
    convert_units: bool = True,
    assay_ids: list[str] | None = None,
    duplicate_treatment="median",
    activity_thresholds: dict[str, float] | None = {
        "active"      : 1000,
        "intermediate": 10000,
        },
    ) -> pd.DataFrame:
    print_low("Starting data preprocessing.")
    print_high("Converting 'standard_value' column to numeric, coercing errors (failing to convert will result in NA).")
    activity_df["standard_value"] = pd.to_numeric(
        arg=activity_df["standard_value"],
        errors="coerce",
        )
    print_high("Filtering out 'standard_value' entries with infinite values.")
    activity_df = activity_df.replace([np.inf, -np.inf], np.nan)
    print_high("Filtering out non-positive 'standard_value' entries.")
    activity_df = activity_df[activity_df["standard_value"] > 0]
    print_high("Dropping rows with NA in key columns.")
    activity_df = activity_df.dropna(subset=["molecule_chembl_id", "canonical_smiles", "standard_value"])
    if assay_ids is not None:
        print_low("Filtering DataFrame by assay ids.")
        print_high(f"Dataframe initial size: {activity_df.shape[0]}")
        activity_df = filter_by_assay(activity_df=activity_df, assay_ids=assay_ids)
        print_high(f"Dataframe filtered size: {activity_df.shape[0]}")
    print_high("Calculating Lipinski descriptors.")
    activity_df = get_lipinski_descriptors(molecules_df=activity_df)
    print_high("Calculating Rule of 5 violations.")
    activity_df = get_ro5_violations(molecules_df=activity_df)
    if convert_units:
        print_high("Converting standard units to Molar (mol/L)")
        activity_df = convert_to_m(molecules_df=activity_df)
    print_high(f"Treating duplicates using '{duplicate_treatment}' method.")
    activity_df = treat_duplicates(
        molecules_df=activity_df,
        method=duplicate_treatment,
        )
    print_high("Normalizing 'standard_value'.")
    activity_df = normalize_value(molecules_df=activity_df)
    print_high("Calculating negative logarithm of 'standard_value'.")
    activity_df = get_neg_log(molecules_df=activity_df)
    print_high("Resetting index.")

    activity_df = activity_df.reset_index(drop=True)
    if activity_thresholds is not None:
        print_high("Assigning bioactivity classes based on thresholds.")
        bioactivity_class = []
        sorted_thresholds = sorted(
            activity_thresholds.items(),
            key=lambda item: item[1],
            )

        for i in activity_df.standard_value:
            value_nm = float(i) * 1e9  # mol/l to n mol/L
            assigned_class = "inactive"
            for class_name, threshold_nM in sorted_thresholds:
                if value_nm <= threshold_nM:
                    assigned_class = class_name
                    break
            bioactivity_class.append(assigned_class)

        activity_df["bioactivity_class"] = bioactivity_class
    print_low("Data preprocessing complete.")

    return activity_df


def scale_features(features, scaler):
    features_scaled = scaler.fit_transform(features)
    features_scaled = pd.DataFrame(features_scaled, index=features.index)
    return features_scaled


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
    treated_molecules_df.loc[:,'standard_value'] = transformed_values
    treated_molecules_df = treated_molecules_df.drop_duplicates(subset='molecule_chembl_id')
    print(f"Filtered DataFrame size: {treated_molecules_df.shape[0]}")
    return treated_molecules_df


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
