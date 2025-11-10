import pandas as pd
from chembl_webresource_client.new_client import new_client

from .utils import print_low, print_high
from .config import settings


def get_activity_data(
    target_chembl_id: str,
    activity_type: str,
    ) -> pd.DataFrame:
    """
    Fetches and processes activity data from the ChEMBL database.

    Filters data for a specific target ChEMBL ID and a standard
    activity type (e.g., "IC50", "Ki").

    Args:
        target_chembl_id (str): The ChEMBL ID of the target (e.g., "CHEMBL203").
        activity_type (str): The type of activity to filter by (e.g., "IC50").

    Returns:
        pd.DataFrame: A DataFrame containing the filtered activity data.
    
    Example:
        ```python
        # Fetch IC50 data for Cyclooxygenase-1 (CHEMBL203)
        target_id = "CHEMBL203"
        activity_type = "IC50"
        
        try:
            activity_df = get_activity_data(target_id, activity_type)
            print(f"Fetched {len(activity_df)} records for {target_id}")
            print(activity_df.head())
        except Exception as e:
            print(f"Error fetching data: {e}")
        ```
    """
    print_low(f"🧪 Fetching '{activity_type}' activity data from ChEMBL for target: {target_chembl_id}")
    activity = new_client.activity  # type: ignore
    activity_query = activity.filter(target_chembl_id=target_chembl_id)
    activity_query = activity_query.filter(standard_type=activity_type)
    activity_df: pd.DataFrame = pd.DataFrame(data=activity_query)
    print_high(f"Fetched {activity_df.shape[0]} records.")
    columns = [
        "molecule_chembl_id",
        "canonical_smiles",
        "molecule_pref_name",
        "target_chembl_id",
        "target_pref_name",
        "assay_chembl_id",
        "assay_description",
        "standard_type",
        "standard_value",
        "standard_units",
        ]
    activity_df = activity_df[columns]
    print_low("✅ Data fetched successfully.")
    return activity_df


def review_assays(
    activity_df: pd.DataFrame,
    max_entries: int = 20,
    assay_keywords: list[str] | None = None,
    exclude_keywords: bool = False,
    inner_join: bool = False,
    ) -> list[str] | None:
    """
    Displays and filters assays from an activity DataFrame.

    Prints a summary of the most common assays and can optionally filter
    this list based on keywords found in the 'assay_description' column.

    Args:
        activity_df (pd.DataFrame): DataFrame containing activity data with
            "assay_chembl_id" and "assay_description" columns.
        max_entries (int, optional): The number of top assays to display in
            the summary. Defaults to 20.
        assay_keywords (list[str] | None, optional): A list of keywords to
            filter assays by. If None, no filtering is done. Defaults to None.
        exclude_keywords (bool, optional): If True, *excludes* assays
            containing the keywords. If False, *includes* them. Defaults to False.
        inner_join (bool, optional): If True, uses "AND" logic for keywords
            (all must be present). If False, uses "OR" logic (any can be
            present). Defaults to False.

    Returns:
        list[str] | None: A list of selected assay ChEMBL IDs, or None if
            no `assay_keywords` are provided.
    
    Example:
        ```python
        # Assuming 'activity_df' is a DataFrame from get_activity_data()
        
        # 1. Just review the top 5 assays without filtering
        print("--- Reviewing Top 5 Assays ---")
        review_assays(activity_df, max_entries=5)

        # 2. Filter for assays containing "HEK293" OR "cell"
        print("\n--- Filtering for 'HEK293' or 'cell' ---")
        include_list = review_assays(
            activity_df,
            max_entries=5,
            assay_keywords=['HEK293', 'cell']
        )
        print(f"Assay IDs to include: {include_list}")

        # 3. Exclude assays containing "mutant" OR "mutated"
        print("\n--- Excluding 'mutant' or 'mutated' ---")
        excluded_list = review_assays(
            activity_df,
            max_entries=5,
            assay_keywords=['mutant', 'mutated'],
            exclude_keywords=True
        )
        print(f"Assay IDs after exclusion: {excluded_list}")
        
        # 4. Filter for assays containing "human" AND "recombinant"
        print("\n--- Filtering for 'human' AND 'recombinant' ---")
        inner_join_list = review_assays(
            activity_df,
            max_entries=5,
            assay_keywords=['human', 'recombinant'],
            inner_join=True
        )
        print(f"Assay IDs with inner join: {inner_join_list}")
        ```
    """
    assay_info = activity_df.loc[:, ["assay_chembl_id", "assay_description"]]
    unique_assays = len(assay_info.value_counts())
    print_low(
        f"Displaying {min(unique_assays, max_entries)} of {unique_assays} total unique assays.",
        )
    print_low("To see more, adjust the 'max_entries' parameter.\n")
    pd.set_option("display.max_rows", max_entries)
    print_low(assay_info.value_counts().head(n=max_entries))

    if assay_keywords is None:
        print_high("No assay_keywords provided, returning None.")
        if settings.verbosity == 0:
            print('No keywords provided. Increase verbosity to review assays.')
        return None
    else:
        if inner_join:
            pattern = "".join([rf"(?=.*{keyword})" for keyword in assay_keywords])
        else:
            pattern = "|".join(assay_keywords)
        print_low("Filtering assays by keywords.")
        print_high(f"Keywords: {assay_keywords}")
        print_high(f"Exclude keywords: {exclude_keywords}")
        print_high(f"Inner join (AND logic): {inner_join}")
        print_high(f"Resulting regex patter: {pattern}")

        mask = assay_info.loc[:, "assay_description"].str.contains(
            pattern,
            case=False,
            na=False,
            )
        if exclude_keywords:
            selected_assays = assay_info[~mask]
        else:
            selected_assays = assay_info[mask]
        unique_selected_assays = len(selected_assays.value_counts())
        print_low(
            f"Displaying {min(unique_selected_assays, max_entries)} of {unique_selected_assays} filtered assays.\n",
            )
        print_low(selected_assays.value_counts().head(n=max_entries))
        selected_id_list = selected_assays.loc[:, "assay_chembl_id"].unique().tolist()  # type: ignore
        return selected_id_list


def filter_by_assay(
    activity_df: pd.DataFrame,
    assay_ids: list[str],
    ) -> pd.DataFrame:
    """
    Filters an activity DataFrame to keep only the specified assay IDs.

    Args:
        activity_df (pd.DataFrame): DataFrame containing ChEMBL activity data.
        assay_ids (list[str]): A list of 'assay_chembl_id' values to keep.

    Returns:
        pd.DataFrame: The filtered DataFrame.
    
    Example:
        ```python
        # Assuming 'activity_df' is a DataFrame from get_activity_data()
        # and 'assay_id_list' is a list of assay IDs from review_assays()
        
        # E.g., assay_id_list = ['CHEMBL123', 'CHEMBL456']
        
        print(f"Original dataframe size: {len(activity_df)}")
        
        filtered_df = filter_by_assay(activity_df, assay_ids=assay_id_list)
        
        print(f"Filtered dataframe size: {len(filtered_df)}")
        ```
    """

    filtered_activity_df = activity_df.loc[
        activity_df["assay_chembl_id"].isin(assay_ids)
    ]
    if filtered_activity_df.empty:
        print("Filtration by assay ids emptied dataframe, returning original dataframe.")
        return activity_df
    else:
        return filtered_activity_df
