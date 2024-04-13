import pytest
import numpy as np

from source.null_imputers.missforest_imputer import MissForestImputer


# Test if output of MissForestImputer does not contain nulls
def test_miss_forest_imputer_no_nulls(acs_income_dataset_categorical_columns_idxs):
    injected_df, categorical_columns_idxs = acs_income_dataset_categorical_columns_idxs
    # Initialize MissForestImputer
    imputer = MissForestImputer()

    # Fit and transform the dataset
    X_imputed = imputer.fit_transform(injected_df, cat_vars=categorical_columns_idxs)

    # Check if there are any nulls in the output
    assert not np.isnan(X_imputed).any(), "Output contains null values"


# Test if MissForestImputer returns same results with the same seed
def test_miss_forest_imputer_same_seed(acs_income_dataset_categorical_columns_idxs, common_seed):
    injected_df, categorical_columns_idxs = acs_income_dataset_categorical_columns_idxs
    
    # Initialize MissForestImputer with seed
    imputer1 = MissForestImputer(seed=common_seed)
    imputer2 = MissForestImputer(seed=common_seed)

    # Fit and transform the sample data with imputer1
    X_imputed1 = imputer1.fit_transform(injected_df, cat_vars=categorical_columns_idxs)

    # Fit and transform the sample data with imputer2
    X_imputed2 = imputer2.fit_transform(injected_df, cat_vars=categorical_columns_idxs)

    # Check if the results are identical
    np.testing.assert_allclose(
        X_imputed1, X_imputed2, 
        atol=1e-9, rtol=1e-9, err_msg="Results from MissForestImputer are not identical"
    )    
    

# Run the tests
if __name__ == "__main__":
    pytest.main([__file__])
