import copy

import pandas as pd

from configs.constants import ACS_INCOME_DATASET
from source.utils.dataframe_utils import encode_cat, decode_cat, encode_cat_with_existing_encoder


def encode_dataset_for_missforest(df, cat_encoders: dict = None, dataset_name: str = None,
                                  categorical_columns_with_nulls: list = None):
    df_enc = copy.deepcopy(df)
    cat_columns = df.select_dtypes(include=['object']).columns

    if dataset_name == ACS_INCOME_DATASET:
        cat_columns_wo_nulls = [c for c in cat_columns if c not in categorical_columns_with_nulls]
        df_enc[cat_columns_wo_nulls] = df_enc[cat_columns_wo_nulls].astype(int)
        cat_encoders = {c: None for c in cat_columns}
    else:
        if cat_encoders is None:
            cat_encoders = dict()
            for c in cat_columns:
                c_enc, encoder = encode_cat(df_enc[c])
                df_enc[c] = c_enc
                cat_encoders[c] = encoder
        else:
            for c in cat_columns:
                df_enc[c] = encode_cat_with_existing_encoder(df_enc[c], cat_encoders[c])

        df_enc[cat_columns] = df_enc[cat_columns].astype('float64')

    # Get indices of categorical columns
    cat_indices = [df_enc.columns.get_loc(col) for col in cat_columns]

    return df_enc, cat_encoders, cat_indices


def encode_dataset_for_nomi(df, cat_encoders: dict = None, dataset_name: str = None):
    df_enc = copy.deepcopy(df)
    cat_columns = df.select_dtypes(include=['object']).columns

    if dataset_name == ACS_INCOME_DATASET:
        df_enc[cat_columns] = df_enc[cat_columns].astype(float)
        cat_encoders = {c: None for c in cat_columns}
    else:
        if cat_encoders is None:
            cat_encoders = dict()
            for c in cat_columns:
                c_enc, encoder = encode_cat(df_enc[c])
                df_enc[c] = c_enc
                cat_encoders[c] = encoder
        else:
            for c in cat_columns:
                df_enc[c] = encode_cat_with_existing_encoder(df_enc[c], cat_encoders[c])

        df_enc[cat_columns] = df_enc[cat_columns].astype('float64')

    # Get indices of categorical columns
    cat_indices = [df_enc.columns.get_loc(col) for col in cat_columns]

    return df_enc, cat_encoders, cat_indices


def encode_dataset_for_gain(X_train: pd.DataFrame, X_tests_lst: list, categorical_columns: list):
    # Combine train and test to find all unique categories
    combined = pd.concat([df[categorical_columns] for df in [X_train] + X_tests_lst])

    # Set all possible categories from the combined data
    for col in categorical_columns:
        all_categories = combined[col].dropna().unique()  # Get all unique categories
        X_train[col] = X_train[col].astype('category')
        X_train[col] = X_train[col].cat.set_categories(all_categories)
        for X_test in X_tests_lst:
            X_test[col] = X_test[col].astype('category')
            X_test[col] = X_test[col].cat.set_categories(all_categories)

    return X_train, X_tests_lst


def decode_dataset_for_gain(X_train: pd.DataFrame, X_tests_lst: list, categorical_columns: list):
    # Convert categorical columns back to string
    for col in categorical_columns:
        X_train[col] = X_train[col].astype(str)
        for X_test in X_tests_lst:
            X_test[col] = X_test[col].astype(str)

    return X_train, X_tests_lst


def decode_dataset_for_missforest(df_enc, cat_encoders, dataset_name: str = None):
    df_dec = copy.deepcopy(df_enc)

    for c in cat_encoders.keys():
        if dataset_name == ACS_INCOME_DATASET:
            df_dec[c] = df_dec[c].astype(int).astype(str)
        else:
            df_dec[c] = decode_cat(df_dec[c], cat_encoders[c])

    return df_dec
