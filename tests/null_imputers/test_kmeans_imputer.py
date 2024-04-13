import pytest
import numpy as np

from source.null_imputers.kmeans_imputer import KMeansImputer


# Test if output of KMeansImputer does not contain nulls
def test_kmeans_imputer_no_nulls(acs_income_dataset_categorical_columns_idxs, common_seed):
    injected_df, categorical_columns_idxs = acs_income_dataset_categorical_columns_idxs
    # Initialize KMeansImputer
    imputer = KMeansImputer(seed=common_seed, n_clusters=3)

    # Fit and transform the dataset
    X_imputed = imputer.fit_transform(injected_df, cat_vars=categorical_columns_idxs)

    # Check if there are any nulls in the output
    assert not np.isnan(X_imputed).any(), "Output contains null values"


# Test if KMeansImputer returns same results with the same seed
def test_kmeans_imputer_same_seed(acs_income_dataset_categorical_columns_idxs, common_seed):
    injected_df, categorical_columns_idxs = acs_income_dataset_categorical_columns_idxs
    
    # Initialize KMeansImputer with seed
    imputer1 = KMeansImputer(seed=common_seed, n_clusters=3)
    imputer2 = KMeansImputer(seed=common_seed, n_clusters=3)

    # Fit and transform the sample data with imputer1
    X_imputed1 = imputer1.fit_transform(injected_df, cat_vars=categorical_columns_idxs)

    # Fit and transform the sample data with imputer2
    X_imputed2 = imputer2.fit_transform(injected_df, cat_vars=categorical_columns_idxs)

    # Check if the results are identical
    np.testing.assert_allclose(
        X_imputed1, X_imputed2, 
        atol=1e-9, rtol=1e-9, err_msg="Results from KMeansImputer are not identical"
    )
    
    
# Run the tests
if __name__ == "__main__":
    pytest.main([__file__])
