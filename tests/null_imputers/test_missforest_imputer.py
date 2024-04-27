import pytest
import numpy as np
import pandas as pd

from source.null_imputers.missforest_imputer import MissForestImputer
from source.utils.dataframe_utils import get_object_columns_indexes


# Test if output of MissForestImputer does not contain nulls
def test_miss_forest_imputer_no_nulls(acs_income_dataset_categorical_columns_idxs, missforest_acs_income_hyperparams):
    injected_df, categorical_columns_idxs = acs_income_dataset_categorical_columns_idxs
    # Initialize MissForestImputer
    imputer = MissForestImputer(hyperparams=missforest_acs_income_hyperparams)

    # Fit and transform the dataset
    X_imputed = imputer.fit_transform(injected_df, cat_vars=categorical_columns_idxs)

    # Check if there are any nulls in the output
    assert not np.isnan(X_imputed).any(), "Output contains null values"


# Test if MissForestImputer returns same results with the same seed
def test_miss_forest_imputer_same_seed(acs_income_dataset_categorical_columns_idxs, common_seed, missforest_acs_income_hyperparams):
    injected_df, categorical_columns_idxs = acs_income_dataset_categorical_columns_idxs
    
    # Initialize MissForestImputer with seed
    imputer1 = MissForestImputer(seed=common_seed, hyperparams=missforest_acs_income_hyperparams)
    imputer2 = MissForestImputer(seed=common_seed, hyperparams=missforest_acs_income_hyperparams)

    # Fit and transform the sample data with imputer1
    X_imputed1 = imputer1.fit_transform(injected_df, cat_vars=categorical_columns_idxs)

    # Fit and transform the sample data with imputer2
    X_imputed2 = imputer2.fit_transform(injected_df, cat_vars=categorical_columns_idxs)

    # Check if the results are identical
    np.testing.assert_allclose(
        X_imputed1, X_imputed2, 
        atol=1e-9, rtol=1e-9, err_msg="Results from MissForestImputer are not identical"
    )    
    
    
def test_miss_forest_imputer_diff_seed(acs_income_dataset_categorical_columns_idxs, common_seed, missforest_acs_income_hyperparams):
    injected_df, categorical_columns_idxs = acs_income_dataset_categorical_columns_idxs
    
    # Initialize MissForestImputer with different seeds
    imputer1 = MissForestImputer(seed=common_seed, hyperparams=missforest_acs_income_hyperparams)
    imputer2 = MissForestImputer(seed=common_seed+1, hyperparams=missforest_acs_income_hyperparams)

    # Fit and transform the sample data with imputer1
    X_imputed1 = imputer1.fit_transform(injected_df, cat_vars=categorical_columns_idxs)

    # Fit and transform the sample data with imputer2
    X_imputed2 = imputer2.fit_transform(injected_df, cat_vars=categorical_columns_idxs)

    # Check if the results are identical
    assert not np.allclose(
        X_imputed1, X_imputed2, 
        atol=1e-9, rtol=1e-9), "Results from MissForestImputer are identical"
    
    
def test_miss_forest_imputer_no_change(acs_income_dataset_categorical_columns_idxs, common_seed, missforest_acs_income_hyperparams):
    injected_df, categorical_columns_idxs = acs_income_dataset_categorical_columns_idxs
    not_null_indices = injected_df.index[injected_df.notnull().all(axis=1)]
    
    # Initialize MissForestImputer
    imputer = MissForestImputer(seed=common_seed, hyperparams=missforest_acs_income_hyperparams)
    
    # Fit and transform the dataset
    X_imputed = imputer.fit_transform(injected_df, cat_vars=categorical_columns_idxs)
    
    X_imputed_df = pd.DataFrame(X_imputed, columns=injected_df.columns, index=injected_df.index)
    X_imputed_df.iloc[:, categorical_columns_idxs] = X_imputed_df.iloc[:, categorical_columns_idxs].astype(int).astype('str')
    
    # check if the rows that had no nulls are the same
    np.testing.assert_array_equal(injected_df.loc[not_null_indices].values, X_imputed_df.loc[not_null_indices].values), "Rows with no nulls have changed"
    

def test_miss_forest_no_change_in_col_with_nulls_for_values_without_nulls(acs_income_dataset_params, common_seed, missforest_acs_income_hyperparams):
    (X_train_val_with_nulls_wo_sensitive_attrs, X_tests_with_nulls_wo_sensitive_attrs_lst,
            train_numerical_null_columns, train_categorical_null_columns,
            numerical_columns_wo_sensitive_attrs, categorical_columns_wo_sensitive_attrs) = acs_income_dataset_params
    not_null_indices = X_train_val_with_nulls_wo_sensitive_attrs.index[X_train_val_with_nulls_wo_sensitive_attrs.notnull().all(axis=1)]
    categorical_columns_idxs = get_object_columns_indexes(X_train_val_with_nulls_wo_sensitive_attrs)
    # Initialize MissForestImputer
    imputer = MissForestImputer(seed=common_seed, hyperparams=missforest_acs_income_hyperparams)
    
    # Fit and transform the dataset
    X_imputed = imputer.fit_transform(X_train_val_with_nulls_wo_sensitive_attrs, cat_vars=categorical_columns_idxs)
    X_imputed_df = pd.DataFrame(X_imputed, columns=X_train_val_with_nulls_wo_sensitive_attrs.columns, index=X_train_val_with_nulls_wo_sensitive_attrs.index)
    X_imputed_df.iloc[:, categorical_columns_idxs] = X_imputed_df.iloc[:, categorical_columns_idxs].astype(int).astype('str')
    
    for col in train_categorical_null_columns:
        np.testing.assert_array_equal(X_train_val_with_nulls_wo_sensitive_attrs.loc[not_null_indices, col].values, X_imputed_df.loc[not_null_indices, col].values, "Rows with no nulls have changed")
        
    for col in train_numerical_null_columns:
        np.testing.assert_array_equal(X_train_val_with_nulls_wo_sensitive_attrs.loc[not_null_indices, col].values, X_imputed_df.loc[not_null_indices, col].values, "Rows with no nulls have changed")


# Run the tests
if __name__ == "__main__":
    pytest.main([__file__])
