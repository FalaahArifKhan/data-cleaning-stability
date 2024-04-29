import pytest

from virny.datasets import ACSIncomeDataset
from source.error_injectors.nulls_injector import NullsInjector


@pytest.fixture(scope="function")
def common_seed():
    return 42


# Fixture to load the dataset
@pytest.fixture(scope="function")
def acs_income_dataloader(common_seed):
    data_loader = ACSIncomeDataset(state=['GA'], year=2018, with_nulls=False,
                                   subsample_size=5_000, subsample_seed=common_seed)

    return data_loader