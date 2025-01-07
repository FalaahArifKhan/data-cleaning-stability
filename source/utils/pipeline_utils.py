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

def generate_types_csv(data: pd.DataFrame, types_file: str):
    """
    Generate a types.csv file based on the dataset's structure.

    Args:
        data (pd.DataFrame): The input dataset.
        types_file (str): Path to save the generated `types.csv`.
    """
    types_data = []

    for col in data.columns:
        if pd.api.types.is_numeric_dtype(data[col]):
            if data[col].min() >= 0 and data[col].apply(float.is_integer).all():
                if len(data[col].unique()) == 2:
                    # Binary categorical column
                    types_data.append({"type": "cat", "dim": 2, "nclass": 2})
                else:
                    # Count column
                    types_data.append({"type": "count", "dim": 1, "nclass": None})
            else:
                # Real-valued column
                types_data.append({"type": "real", "dim": 1, "nclass": None})
        elif pd.api.types.is_categorical_dtype(data[col]) or pd.api.types.is_object_dtype(data[col]):
            unique_classes = data[col].nunique()
            types_data.append({"type": "cat", "dim": unique_classes, "nclass": unique_classes})
        elif pd.api.types.is_integer_dtype(data[col]):
            # Ordinal column
            unique_classes = data[col].nunique()
            types_data.append({"type": "ordinal", "dim": unique_classes, "nclass": unique_classes})
        else:
            raise ValueError(f"Unsupported column type for column '{col}'.")

    # Write to CSV
    with open(types_file, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["type", "dim", "nclass"])
        writer.writeheader()
        writer.writerows(types_data)

    print(f"`types.csv` generated and saved to {types_file}.")
