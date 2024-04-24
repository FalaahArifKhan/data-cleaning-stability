import pytest

from virny.datasets import ACSIncomeDataset
from sklearn.model_selection import train_test_split

from configs.datasets_config import ACS_INCOME_DATASET
from configs.null_imputers_config import NULL_IMPUTERS_HYPERPARAMS
from configs.constants import ErrorRepairMethod, ErrorInjectionStrategy
from source.custom_classes.benchmark import Benchmark
from source.error_injectors.nulls_injector import NullsInjector
from source.utils.dataframe_utils import get_object_columns_indexes


@pytest.fixture(scope="function")
def missforest_acs_income_hyperparams():
    hyperparams = NULL_IMPUTERS_HYPERPARAMS.get(ErrorRepairMethod.miss_forest.value).get(ACS_INCOME_DATASET).get(ErrorInjectionStrategy.mcar.value)
    return hyperparams


@pytest.fixture(scope="function")
def kmeans_acs_income_hyperparams():
    hyperparams = NULL_IMPUTERS_HYPERPARAMS.get(ErrorRepairMethod.k_means_clustering.value).get(ACS_INCOME_DATASET).get(ErrorInjectionStrategy.mcar.value)
    return hyperparams


@pytest.fixture(scope="function")
def common_seed():
    return 42


@pytest.fixture(scope="function")
def mcar_evaluation_scenario():
    return 'exp3_mcar3'


# Fixture to load the dataset
@pytest.fixture(scope="function")
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


@pytest.fixture(scope="function")
def acs_income_dataset_params(common_seed, mcar_evaluation_scenario):
    experiment_seed = common_seed
    evaluation_scenario = mcar_evaluation_scenario

    benchmark = Benchmark(dataset_name=ACS_INCOME_DATASET,
                          null_imputers=[],
                          model_names=[])
    benchmark.init_data_loader = ACSIncomeDataset(state=['GA'], year=2018, with_nulls=False,
                                                  subsample_size=1_000, subsample_seed=experiment_seed)

    # Split the dataset
    X_train_val, X_test, y_train_val, y_test = train_test_split(benchmark.init_data_loader.X_data,
                                                                benchmark.init_data_loader.y_data,
                                                                test_size=benchmark.test_set_fraction,
                                                                random_state=experiment_seed)

    # Inject nulls not into sensitive attributes
    X_train_val_with_nulls, X_tests_with_nulls_lst = benchmark._inject_nulls(X_train_val=X_train_val,
                                                                             X_test=X_test,
                                                                             evaluation_scenario=evaluation_scenario,
                                                                             experiment_seed=experiment_seed)

    # Remove sensitive attributes from train and test sets with nulls to avoid their usage during imputation
    (X_train_val_with_nulls_wo_sensitive_attrs,
     X_tests_with_nulls_wo_sensitive_attrs_lst,
     numerical_columns_wo_sensitive_attrs,
     categorical_columns_wo_sensitive_attrs) = benchmark._remove_sensitive_attrs(X_train_val=X_train_val_with_nulls,
                                                                                 X_tests_lst=X_tests_with_nulls_lst,
                                                                                 data_loader=benchmark.init_data_loader)

    train_set_cols_with_nulls = X_train_val_with_nulls_wo_sensitive_attrs.columns[X_train_val_with_nulls_wo_sensitive_attrs.isna().any()].tolist()
    train_numerical_null_columns = list(set(train_set_cols_with_nulls).intersection(numerical_columns_wo_sensitive_attrs))
    train_categorical_null_columns = list(set(train_set_cols_with_nulls).intersection(categorical_columns_wo_sensitive_attrs))

    return (X_train_val_with_nulls_wo_sensitive_attrs, X_tests_with_nulls_wo_sensitive_attrs_lst,
            train_numerical_null_columns, train_categorical_null_columns,
            numerical_columns_wo_sensitive_attrs, categorical_columns_wo_sensitive_attrs)
