import copy
import pandas as pd
import csv
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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


def encode_dataset_for_mnar_pvae(df, cat_encoders: dict = None, dataset_name: str = None, scaler = None):
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

    # Normalize features
    if scaler is None:
        scaler = StandardScaler()
    normalized_data = scaler.fit_transform(df_enc)
    df_enc = pd.DataFrame(normalized_data, columns=df.columns, index=df.index)

    return df_enc, cat_encoders, scaler


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


def decode_dataset_for_mnar_pvae(df_enc, cat_encoders, dataset_name: str = None, scaler = None):
    df_dec = copy.deepcopy(df_enc)
    denormalized_data = scaler.inverse_transform(df_dec) # Denormalize features
    df_dec = pd.DataFrame(denormalized_data, columns=df_enc.columns, index=df_enc.index)

    for c in cat_encoders.keys():
        if dataset_name == ACS_INCOME_DATASET:
            df_dec[c] = df_dec[c].round().astype(int).astype(str)
        else:
            df_dec[c] = decode_cat(df_dec[c].round().astype(int), cat_encoders[c])

    return df_dec


def onehot_encode_dataset(df, encoder=None):
    df_enc = copy.deepcopy(df)
    cat_columns = df.select_dtypes(include=['object']).columns
    num_columns = [col for col in df.columns if col not in cat_columns]

    if encoder:
        encoded_array = encoder.transform(df_enc[cat_columns])
    else:
        encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        encoded_array = encoder.fit_transform(df_enc[cat_columns])

    df_enc_cat = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out(cat_columns))
    df_enc = pd.concat([df_enc[num_columns], df_enc_cat], axis=1)
    return df_enc, encoder, cat_columns


def onehot_decode_dataset(df, encoder, init_cat_columns):
    df_dec = copy.deepcopy(df)
    onehot_cat_columns = encoder.get_feature_names_out(init_cat_columns)
    num_columns = [col for col in df.columns if col not in onehot_cat_columns]

    reversed_array = encoder.inverse_transform(df_dec[onehot_cat_columns].to_numpy())
    df_dec_cat = pd.DataFrame(reversed_array, columns=init_cat_columns)
    df_dec = pd.concat([df_dec[num_columns], df_dec_cat], axis=1)
    return df_dec

def generate_types_csv(df: pd.DataFrame, output_path: str):
    """
    Example stub that writes a minimal 'types.csv' given a DataFrame.
    You must adapt it to match your columns, data types, and dimension counting.
    E.g. for numeric columns => type='real', dim=1,
         for categorical => type='cat', dim=#unique_categories, etc.
    """
    import csv

    # Simple guess: if it's numeric => 'real', dim=1;
    # if it's object/category => 'cat', dim=number of unique non-null categories
    rows = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            # e.g., treat as real
            rows.append({'name': col, 'type': 'pos', 'dim': 1})
        else:
            # treat as cat
            unique_vals = df[col].dropna().unique()
            cat_dim = len(unique_vals) if len(unique_vals) > 1 else 2
            rows.append({'name': col, 'type': 'cat', 'dim': cat_dim})

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['name', 'type', 'dim']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def generate_types_dict(df: pd.DataFrame):
    """
    Example stub that writes a minimal 'types.csv' given a DataFrame.
    You must adapt it to match your columns, data types, and dimension counting.
    E.g. for numeric columns => type='real', dim=1,
         for categorical => type='cat', dim=#unique_categories, etc.
    """
    # Simple guess: if it's numeric => 'real', dim=1;
    # if it's object/category => 'cat', dim=number of unique non-null categories
    rows = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            # e.g., treat as real
            rows.append({'name': col, 'type': 'real', 'dim': 1})
        else:
            # treat as cat
            unique_vals = df[col].dropna().unique()
            cat_dim = len(unique_vals) if len(unique_vals) > 1 else 2
            rows.append({'name': col, 'type': 'cat', 'dim': cat_dim})

    return rows

   
def parse_types_csv_file(types_file):
    """
    Utility to parse 'types.csv' into a Python list of dicts.
    Each row in the CSV is typically:
       name,type,dim
    Example row:
       col0,cat,3
    This returns:
       [{'name': 'col0', 'type': 'cat', 'dim': 3}, ...]
    """
    types_dict = []
    with open(types_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert 'dim' to int
            row['dim'] = int(row['dim'])
            types_dict.append(row)
    return types_dict
