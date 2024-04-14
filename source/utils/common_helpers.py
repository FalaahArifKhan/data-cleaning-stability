import hashlib
import secrets
import base64
from virny.custom_classes.base_dataset import BaseFlowDataset


def generate_guid(ordered_hierarchy_lst: list):
    identifier = '|'.join([str(val) for val in ordered_hierarchy_lst])
    return hashlib.md5(identifier.encode()).hexdigest()


def generate_base64_hash(length=8):
    # Generate random bytes. Since Base64 encodes each 3 bytes into 4 characters, calculate bytes needed.
    bytes_needed = (length * 3) // 4
    random_bytes = secrets.token_bytes(bytes_needed)

    # Encode bytes in base64 and decode to utf-8 string
    random_hash = base64.urlsafe_b64encode(random_bytes).decode('utf-8')

    # Return the required length
    return random_hash[:length]


def create_virny_base_flow_dataset(data_loader, dataset_sensitive_attrs,
                                   X_train_val_wo_sensitive_attrs, X_test_wo_sensitive_attrs,
                                   y_train_val, y_test, numerical_columns_wo_sensitive_attrs,
                                   categorical_columns_wo_sensitive_attrs):
    # Create a dataframe with sensitive attributes and initial dataset indexes
    sensitive_attrs_df = data_loader.full_df[dataset_sensitive_attrs]

    # Ensure correctness of indexes in X and sensitive_attrs sets
    assert X_train_val_wo_sensitive_attrs.index.isin(sensitive_attrs_df.index).all(), \
        "Not all indexes of X_train_val_wo_sensitive_attrs are present in sensitive_attrs_df"
    assert X_test_wo_sensitive_attrs.index.isin(sensitive_attrs_df.index).all(), \
        "Not all indexes of X_test_wo_sensitive_attrs are present in sensitive_attrs_df"

    # Ensure correctness of indexes in X and y sets
    assert X_train_val_wo_sensitive_attrs.index.equals(y_train_val.index) is True, \
        "Indexes of X_train_val_wo_sensitive_attrs and y_train_val are different"
    assert X_test_wo_sensitive_attrs.index.equals(y_test.index) is True, \
        "Indexes of X_test_wo_sensitive_attrs and y_test are different"

    return BaseFlowDataset(init_features_df=sensitive_attrs_df,  # keep only sensitive attributes with original indexes to compute group metrics
                           X_train_val=X_train_val_wo_sensitive_attrs,
                           X_test=X_test_wo_sensitive_attrs,
                           y_train_val=y_train_val,
                           y_test=y_test,
                           target=data_loader.target,
                           numerical_columns=numerical_columns_wo_sensitive_attrs,
                           categorical_columns=categorical_columns_wo_sensitive_attrs)
