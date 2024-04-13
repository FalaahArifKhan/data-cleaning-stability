import pandas as pd


def compare_dfs(df1, df2):
    # Check shape
    if not df1.shape == df2.shape:
        return False

    # Check column names
    if not sorted(df1.columns.tolist()) == sorted(df2.columns.tolist()):
        return False

    # Check values
    if not df1.equals(df2):
        return False

    return True


def get_df_condition(df: pd.DataFrame, condition_col: str, condition_val, include_val: bool):
    if isinstance(condition_val, list):
        df_condition = df[condition_col].isin(condition_val) if include_val else ~df[condition_col].isin(condition_val)
    else:
        df_condition = df[condition_col] == condition_val if include_val else df[condition_col] != condition_val

    return df_condition


def assert_nested_dicts_equal(dict1: dict, dict2: dict, assert_msg: str):
    """
    Recursively compares two nested dictionaries for equality.
    """
    # Check if both arguments are dictionaries
    assert isinstance(dict1, dict) and isinstance(dict2, dict)

    # Check if both dictionaries have the same keys
    assert sorted(dict1.keys()) == sorted(dict2.keys())

    # Iterate through the keys in dict1
    for key in dict1:
        # Recursively compare the values for nested dictionaries
        if isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
            assert_nested_dicts_equal(dict1[key], dict2[key], assert_msg)
        else:
            # Compare the values for non-dictionary values
            assert dict1[key] == dict2[key], assert_msg
