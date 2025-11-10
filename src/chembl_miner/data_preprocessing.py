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
    """
    Runs the full preprocessing pipeline on a ChEMBL activity DataFrame.

    Steps include:
    1.  Convert 'standard_value' to numeric and remove NaNs/Infs.
    2.  Filter out non-positive values.
    3.  Optionally filter by provided `assay_ids`.
    4.  Calculate Lipinski descriptors and Rule of 5 violations.
    5.  Optionally convert all activity units to Molar (M).
    6.  Treat duplicate molecule entries using the specified method.
    7.  Normalize 'standard_value' (cap at 0.1 M).
    8.  Calculate 'neg_log_value' (e.g., pIC50).
    9.  Optionally assign bioactivity classes based on thresholds.

    Args:
        activity_df (pd.DataFrame): The raw activity data from ChEMBL.
        convert_units (bool, optional): Whether to convert units to Molar.
            Defaults to True.
        assay_ids (list[str] | None, optional): A list of assay ChEMBL IDs to
            keep. If None, all assays are kept. Defaults to None.
        duplicate_treatment (str, optional): Method to handle duplicates
            ('median', 'mean', 'max', 'min'). Defaults to "median".
        activity_thresholds (dict[str, float] | None, optional): Thresholds
            (in nM) for classifying bioactivity. The keys are class names
            (e.g., "active") and values are the upper limit (inclusive).
            Set to None to skip classification. Defaults to
            {"active": 1000, "intermediate": 10000}.

    Returns:
        pd.DataFrame: The preprocessed DataFrame.
    
    Example:
        ```python
        # Create a sample activity DataFrame
        data = {
            'molecule_chembl_id': ['CHEMBL1', 'CHEMBL1', 'CHEMBL2', 'CHEMBL3'],
            'canonical_smiles': ['CCC', 'CCC', 'CCO', 'CCN'], # SMILES for Lipinski
            'standard_value': [500, 600, 2000, 50000],
            'standard_units': ['nM', 'nM', 'nM', 'nM'],
            'assay_chembl_id': ['A1', 'A1', 'A2', 'A1']
        }
        activity_df = pd.DataFrame(data)
        
        # Run preprocessing, averaging duplicates
        processed_df = preprocess_data(
            activity_df,
            duplicate_treatment='mean',
            activity_thresholds={"active": 1000, "intermediate": 10000}
        )
        
        print(processed_df[['molecule_chembl_id', 'neg_log_value', 'bioactivity_class']])
        # Output:
        #   molecule_chembl_id  neg_log_value bioactivity_class
        # 0            CHEMBL1       6.259637            active
        # 1            CHEMBL2       5.698970      intermediate
        # 2            CHEMBL3       4.301030          inactive
        ```
    """
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
    """
    Fits and transforms features using a scikit-learn scaler.

    Args:
        features (pd.DataFrame): DataFrame of features to scale.
        scaler (sklearn.preprocessing.Scaler): An instance of a scaler
            (e.g., StandardScaler, MinMaxScaler).

    Returns:
        pd.DataFrame: The scaled features, preserving the original index.
    
    Example:
        ```python
        from sklearn.preprocessing import StandardScaler
        
        features_df = pd.DataFrame(
            {'MW': [300, 450, 500], 'LogP': [3.1, 4.5, 5.2]},
            index=['mol1', 'mol2', 'mol3']
        )
        scaler = StandardScaler()
        
        scaled_features = scale_features(features_df, scaler)
        print(scaled_features)
        #            MW      LogP
        # mol1 -1.240347 -1.341198
        # mol2  0.155043  0.149022
        # mol3  1.085304  1.192176
        ```
    """
    features_scaled = scaler.fit_transform(features)
    features_scaled = pd.DataFrame(features_scaled, index=features.index)
    return features_scaled


def remove_low_variance_columns(input_data, threshold=0.1):
    """
    Removes feature columns with near-zero variance.

    Args:
        input_data (pd.DataFrame): The feature DataFrame.
        threshold (float, optional): The variance threshold. Features with
            variance below this value will be removed. Defaults to 0.1.

    Returns:
        pd.DataFrame: The DataFrame with low-variance columns removed.
    
    Example:
        ```python
        features_df = pd.DataFrame({
            'feat1': [10, 12, 11, 10],      # Variance = 0.916
            'feat2': [1, 1, 1, 1],          # Variance = 0.0
            'feat3': [0.1, 0.1, 0.1, 0.2]   # Variance = 0.0025
        })
        
        # Will remove 'feat2' and 'feat3' (variance < 0.1)
        filtered_df = remove_low_variance_columns(features_df, threshold=0.1)
        
        print(filtered_df.columns) 
        # Output: Index(['feat1'], dtype='object')
        ```
    """
    selection = VarianceThreshold(threshold)
    selection.fit(input_data)
    return input_data[input_data.columns[selection.get_support(indices=True)]]


def normalize_value(molecules_df):
    """
    Caps the 'standard_value' column at 0.1 for transformation by negative logarithm.

    Values greater than 0.1 are set to 0.1.

    This function is called by preprocess_data() function.

    Args:
        molecules_df (pd.DataFrame): DataFrame with a 'standard_value' column.

    Returns:
        pd.DataFrame: The DataFrame with capped values.
    
    Example:
        ```python
        data = {'standard_value': [0.05, 0.1, 0.5, 1.2]}
        df = pd.DataFrame(data)
        
        normalized_df = normalize_value(df)
        
        # print(normalized_df['standard_value'].tolist())
        # Output: [0.05, 0.1, 0.1, 0.1]
        ```
    """
    norm = []
    molecules_df_norm = molecules_df

    for i in molecules_df_norm['standard_value']:
        if float(i) > 0.1:
            i = 0.1
        norm.append(i)

    molecules_df_norm['standard_value'] = norm
    return molecules_df_norm


def get_neg_log(molecules_df):
    """
    Calculates the negative base-10 logarithm of the 'standard_value' column.

    Creates a new column named 'neg_log_value'.

    This function is called by preprocess_data() function.

    Args:
        molecules_df (pd.DataFrame): DataFrame with a 'standard_value' column
            (assumed to be in Molar).

    Returns:
        pd.DataFrame: The DataFrame with the new 'neg_log_value' column.
    
    Example:
        ```python
        # Values are assumed to be in Molar (M)
        data = {'standard_value': [1e-9, 1e-7, 5e-8]} # 1nM, 100nM, 50nM
        df = pd.DataFrame(data)
        
        df_with_neg_log = get_neg_log(df)
        
        print(df_with_neg_log['neg_log_value'].tolist())
        # Output: [9.0, 7.0, 7.301029995663981]
        ```
    """
    neg_log = []
    molecules_df_neg_log = molecules_df

    for i in molecules_df_neg_log['standard_value']:
        i = float(i)
        neg_log.append(-np.log10(i))

    molecules_df_neg_log['neg_log_value'] = neg_log
    return molecules_df_neg_log


def treat_duplicates(molecules_df, method: str = 'median') -> pd.DataFrame:
    """
    Resolves duplicate molecule entries by aggregating their 'standard_value'.

    Groups by 'molecule_chembl_id', applies the specified aggregation method
    (e.g., 'median') to the 'standard_value', and then drops duplicate IDs.

    This function is called by preprocess_data() function.

    Args:
        molecules_df (pd.DataFrame): DataFrame containing molecule data.
        method (str, optional): The aggregation method to apply. One of
            ['median', 'mean', 'max', 'min']. Defaults to 'median'.

    Returns:
        pd.DataFrame: A DataFrame with duplicate molecules resolved.
    
    Example:
        ```python
        data = {
            'molecule_chembl_id': ['A', 'B', 'A', 'C', 'A'],
            'standard_value': [100, 500, 200, 1000, 800]
        }
        df = pd.DataFrame(data)
        
        # Use 'mean' to average values for 'A' (100 + 200 + 800) / 3 = 366.7
        treated_df_mean = treat_duplicates(df, method='mean')
        
        print(treated_df_mean)
        #   molecule_chembl_id  standard_value
        # 0                    A             366.7
        # 1                    B             500.0
        # 3                    C            1000.0

        # Use 'median' (default) for 'A' (100, 200, 800) -> 200
        treated_df_median = treat_duplicates(df, method='median')
        print(treated_df_median)
        #   molecule_chembl_id  standard_value
        # 0                    A             200.0
        # 1                    B             500.0
        # 3                    C            1000.0
        ```
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
    """
    Converts 'standard_value' from various units to Molar (M).

    Handles 'nM', 'uM', 'mM', 'M', and 'ug.mL-1' / 'ug ml-1' (requires 'MW'
    column for the latter).

    This function is called by preprocess_data() function.

    Args:
        molecules_df (pd.DataFrame): DataFrame with 'standard_value',
            'standard_units', and (if needed) 'MW' columns.

    Returns:
        pd.DataFrame: A DataFrame where all 'standard_value' entries are in
            Molar and 'standard_units' is set to 'M'.
    
    Example:
        ```python
        data = {
            'standard_value': [100, 50, 2, 80],
            'standard_units': ['nM', 'uM', 'mM', 'ug.mL-1'],
            'MW': [400, 400, 400, 400] # MW is 400 g/mol, needed for ug.mL-1
        }
        df = pd.DataFrame(data)
        
        converted_df = convert_to_m(df)
        
        print(converted_df[['standard_value', 'standard_units']])
        #   standard_value standard_units
        # 0    1.000000e-07              M
        # 1    5.000000e-05              M
        # 2    2.000000e-03              M
        # 3    2.000000e-04              M
        ```
    """

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
    """
    Calculates the number of Lipinski's Rule of 5 (Ro5) violations.

    Checks for:
    - MW >= 500
    - LogP >= 5
    - NumHDonors >= 5
    - NumHAcceptors >= 10

    Creates a new column 'Ro5Violations'. Requires Lipinski descriptors
    ('MW', 'LogP', 'NumHDonors', 'NumHAcceptors') to be present.

    This function is called by preprocess_data() function.

    Args:
        molecules_df (pd.DataFrame): DataFrame with Lipinski descriptors.

    Returns:
        pd.DataFrame: The DataFrame with the new 'Ro5Violations' column.
    
    Example:
        ```python
        data = {
            'MW':            [300, 550, 400, 510],
            'LogP':          [4.0, 4.8, 5.2, 5.5],
            'NumHDonors':    [2,   4,   6,   7],
            'NumHAcceptors': [8,   9,   8,  11]
        }
        df = pd.DataFrame(data, index=['mol1', 'mol2', 'mol3', 'mol4'])
        
        df_with_violations = get_ro5_violations(df)
        
        # print(df_with_violations['Ro5Violations'])
        # mol1    0
        # mol2    1
        # mol3    2
        # mol4    4
        # Name: Ro5Violations, dtype: int64
        ```
    """
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
