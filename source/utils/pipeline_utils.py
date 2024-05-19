import copy

from configs.constants import ACS_INCOME_DATASET
from source.utils.dataframe_utils import encode_cat, decode_cat, encode_cat_with_existing_encoder


def encode_dataset_for_missforest(df, cat_encoders: dict = None, dataset_name: str = None,
                                  categorical_columns_with_nulls: list = None):
    df_enc = copy.deepcopy(df)
    cat_columns = df.select_dtypes(include=['object']).columns

    if dataset_name == ACS_INCOME_DATASET:
        cat_columns_wo_nulls = [c for c in cat_columns if c not in categorical_columns_with_nulls]
        df_enc[cat_columns_wo_nulls] = df_enc[cat_columns_wo_nulls].astype(int)
        
        if cat_encoders is None:
            cat_encoders = dict()
            for c in categorical_columns_with_nulls:
                c_enc, encoder = encode_cat(df_enc[c])
                df_enc[c] = c_enc
                cat_encoders[c] = encoder
        else:
            for c in categorical_columns_with_nulls:
                df_enc[c] = encode_cat_with_existing_encoder(df_enc[c], cat_encoders[c])
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

    # Get indices of categorical columns
    cat_indices = [df_enc.columns.get_loc(col) for col in cat_columns]

    return df_enc, cat_encoders, cat_indices


def decode_dataset_for_missforest(df_enc, cat_encoders, dataset_name: str = None):
    df_dec = copy.deepcopy(df_enc)

    for c in cat_encoders.keys():
        if dataset_name == ACS_INCOME_DATASET:
            df_dec[c] = df_dec[c].astype(int).astype(str)
        else:
            df_dec[c] = decode_cat(df_dec[c], cat_encoders[c])

    return df_dec
