import pandas as pd
import pandas.testing as pdt

from virny.custom_classes.base_dataset import BaseFlowDataset


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


def get_condition_by_symbol(df, condition_col, symbol, condition_val):
    if symbol == 'ge':
        return df[condition_col] >= condition_val
    elif symbol == 'gt':
        return df[condition_col] > condition_val
    elif symbol == 'le':
        return df[condition_col] <= condition_val
    elif symbol == 'lt':
        return df[condition_col] < condition_val


def get_df_condition(df: pd.DataFrame, condition_col: str, condition_val, include_val: bool):
    if isinstance(condition_val, list):
        df_condition = df[condition_col].isin(condition_val) if include_val else ~df[condition_col].isin(condition_val)

    elif isinstance(condition_val, dict):
        # Validate condition
        symbol_counts = {'g': 0, 'l': 0}
        for key in condition_val.keys():
            if key not in ('ge', 'gt', 'le', 'lt'):
                raise ValueError(f"Condition symbol {key} is not in ('ge', 'gt', 'le', 'lt')")
            symbol_counts[key[0]] += 1

        if symbol_counts['g'] > 1 or symbol_counts['l'] > 1:
            raise ValueError(f"Condition should not include more than one greater symbol "
                             f"or more than one less symbol")

        df_condition = None
        for symbol in condition_val.keys():
            val = condition_val[symbol]
            cur_df_condition = get_condition_by_symbol(df=df,
                                                       condition_col=condition_col,
                                                       symbol=symbol,
                                                       condition_val=val)
            if df_condition is None:
                df_condition = cur_df_condition
            else:
                df_condition &= cur_df_condition

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


def compare_base_flow_datasets(dataset1: BaseFlowDataset, dataset2: BaseFlowDataset):
    # Assert equality for DataFrames
    pdt.assert_frame_equal(dataset1.init_sensitive_attrs_df, dataset2.init_sensitive_attrs_df)
    pdt.assert_frame_equal(dataset1.X_train_val, dataset2.X_train_val)
    pdt.assert_frame_equal(dataset1.X_test, dataset2.X_test)
    if isinstance(dataset1.y_train_val, pd.Series):
        pdt.assert_series_equal(dataset1.y_train_val, dataset2.y_train_val)
    else:
        pdt.assert_frame_equal(dataset1.y_train_val, dataset2.y_train_val)
    pdt.assert_series_equal(dataset1.y_test, dataset2.y_test)

    # Assert equality for lists
    assert dataset1.numerical_columns == dataset2.numerical_columns, "Numerical columns do not match"
    assert dataset1.categorical_columns == dataset2.categorical_columns, "Categorical columns do not match"

    # Assert equality for strings
    assert dataset1.target == dataset2.target, "Targets do not match"
