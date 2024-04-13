import pytest
from virny.datasets import ACSIncomeDataset

from source.error_injectors.nulls_injector import NullsInjector
from source.utils.dataframe_utils import get_object_columns_indexes


@pytest.fixture(scope="session")
def common_seed():
    return 42


# Fixture to load the dataset
@pytest.fixture(scope="session")
def acs_income_dataset_categorical_columns_idxs(common_seed):
    data_loader = ACSIncomeDataset(state=['GA'], year=2018, with_nulls=False,
                                   subsample_size=5_000, subsample_seed=common_seed)
    mcar_injector = NullsInjector(seed=common_seed,
                                  strategy='MCAR',
                                  columns_with_nulls=["AGEP", "SCHL", "MAR"],
                                  null_percentage=0.3)
    injected_df = mcar_injector.transform(data_loader.X_data)

    categorical_columns_idxs = get_object_columns_indexes(injected_df)

    return injected_df, categorical_columns_idxs
