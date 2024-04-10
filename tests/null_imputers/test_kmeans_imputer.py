import pytest
import numpy as np
from virny.datasets import ACSIncomeDataset

from source.null_imputers.kmeans_imputer import KMeansImputer
from source.error_injectors.nulls_injector import NullsInjector
from source.utils.dataframe_utils import get_object_columns_indexes

SEED = 42


# Fixture to load the dataset
@pytest.fixture(scope="module")
def acs_income_dataset_categorical_columns_idxs():
    data_loader = ACSIncomeDataset(state=['GA'], year=2018, with_nulls=False,
                                   subsample_size=5_000, subsample_seed=SEED)
    mcar_injector = NullsInjector(seed=SEED, strategy='MCAR', columns_with_nulls=["AGEP", "SCHL", "MAR"], null_percentage=0.3)
    injected_df = mcar_injector.transform(data_loader.X_data)
    
    categorical_columns_idxs = get_object_columns_indexes(injected_df)

    return injected_df, categorical_columns_idxs


# Test if output of KMeansImputer does not contain nulls
def test_kmeans_imputer_no_nulls(acs_income_dataset_categorical_columns_idxs):
    injected_df, categorical_columns_idxs = acs_income_dataset_categorical_columns_idxs
    # Initialize KMeansImputer
    imputer = KMeansImputer(seed=SEED, n_clusters=3)

    # Fit and transform the dataset
    X_imputed = imputer.fit_transform(injected_df, cat_vars=categorical_columns_idxs)

    # Check if there are any nulls in the output
    assert not np.isnan(X_imputed).any(), "Output contains null values"


# Test if KMeansImputer returns same results with the same seed
def test_kmeans_imputer_same_seed(acs_income_dataset_categorical_columns_idxs):
    injected_df, categorical_columns_idxs = acs_income_dataset_categorical_columns_idxs
    
    # Initialize KMeansImputer with seed
    imputer1 = KMeansImputer(seed=SEED, n_clusters=3)
    imputer2 = KMeansImputer(seed=SEED, n_clusters=3)

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