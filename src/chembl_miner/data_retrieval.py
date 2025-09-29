import pandas as pd
from chembl_webresource_client.new_client import new_client

from .utils import print_low, print_high
from .config import settings


def get_activity_data(
    target_chembl_id: str,
    activity_type: str,
    ) -> pd.DataFrame:
    """
    Fetches and processes activity data from the ChEMBL database for a given target ID and activity type.
    Args:
        target_chembl_id (str): The ChEMBL ID of the target.
        activity_type (str): The type of activity to filter results by.
    Returns:
        dataset (pd.DataFrame): A dataframe object containing the processed dataset.
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
    Displays and filter assays from an activity DataFrame.

    Can either include or exclude assays based on keywords found in the
    'assay_description' column.

    Args:
        activity_df (pd.DataFrame): DataFrame containing activity data with
                                    "assay_chembl_id" and "assay_description" columns.
        max_entries (int): The number of top assays to display. Defaults to 20.
        assay_keywords (list[str] | None): A list of keywords to filter
                                           assays by. Defaults to None.
        exclude_keywords (bool):Use the keywords for exclusion instead of inclusion.
                                Defaults to False.
        inner_join (bool): Use inner join instead of outer (AND instead of OR).
                           Defaults to False.

    Returns:
        List[str] | None: A list of selected assay ChEMBL IDs, or None if no
                          keywords are provided or an error occurs.
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
    Filters an activity DataFrame by 'assay_description' column using provided assay_ids.
    Args:
        activity_df (pd.DataFrame): DataFrame containing ChEMBL activity data.
        assay_ids (list[str]): list of assay_chembl_ids obtained from the review_assays function.

    Returns:
        pd.DataFrame:
    """

    filtered_activity_df = activity_df.loc[
        activity_df["assay_chembl_id"].isin(assay_ids)
    ]
    if filtered_activity_df.empty:
        print("Filtration by assay ids emptied dataframe, returning original dataframe.")
        return activity_df
    else:
        return filtered_activity_df
