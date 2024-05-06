import copy

from source.utils.dataframe_utils import encode_cat, decode_cat, encode_cat_with_existing_encoder


def encode_dataset_for_missforest(df, cat_encoders: dict = None):
    df_enc = copy.deepcopy(df)
    cat_columns = df.select_dtypes(include=['object']).columns

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


def decode_dataset_for_missforest(df_enc, cat_encoders):
    df_dec = copy.deepcopy(df_enc)

    for c in cat_encoders.keys():
        df_dec[c] = decode_cat(df_dec[c], cat_encoders[c])

    return df_dec
