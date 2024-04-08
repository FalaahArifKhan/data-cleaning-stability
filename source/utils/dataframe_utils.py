import pandas as pd


def get_columns_sorted_by_nulls(df):
    # Calculate the number of null values in each column
    null_counts = df.isna().sum()

    # Sort columns based on the number of null values
    sorted_columns = null_counts.sort_values(ascending=True)

    # Get the column names as a list
    sorted_columns_names = sorted_columns.index.tolist()

    return sorted_columns_names
