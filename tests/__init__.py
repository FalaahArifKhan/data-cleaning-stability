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
