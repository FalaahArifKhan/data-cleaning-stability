import copy
import pytest
from sklearn.model_selection import train_test_split

from tests import compare_dfs
from source.custom_classes.benchmark import Benchmark
from configs.constants import ACS_INCOME_DATASET, ErrorRepairMethod, MLModels, EXPERIMENT_RUN_SEEDS


@pytest.fixture(scope='module')
def benchmark():
    benchmark = Benchmark(dataset_name=ACS_INCOME_DATASET,
                          null_imputers=[ErrorRepairMethod.median_mode.value],
                          model_names=[MLModels.lr_clf.value])

    return benchmark

# ====================================================================
# Test error injection
# ====================================================================
def test_inject_nulls_should_be_the_same_mcar_train_sets_with_nulls(benchmark):
    evaluation_scenarios = ['mcar_mcar2', 'mcar_mar2', 'mcar_mnar2']
    experiment_seed = 100
    dataset_pairs_with_nulls = []
    for evaluation_scenario in evaluation_scenarios:
        data_loader = copy.deepcopy(benchmark.init_data_loader)

        # Split and preprocess the dataset
        X_train_val, X_test, y_train_val, y_test = train_test_split(data_loader.X_data,
                                                                    data_loader.y_data,
                                                                    test_size=benchmark.test_set_fraction,
                                                                    random_state=experiment_seed)
        # Inject nulls
        X_train_val_with_nulls, X_test_with_nulls = benchmark._inject_nulls(X_train_val=X_train_val,
                                                                            X_test=X_test,
                                                                            evaluation_scenario=evaluation_scenario,
                                                                            experiment_seed=experiment_seed)

        dataset_pairs_with_nulls.append((X_train_val_with_nulls, X_test_with_nulls))

    assert compare_dfs(dataset_pairs_with_nulls[0][0], dataset_pairs_with_nulls[1][0])

    assert compare_dfs(dataset_pairs_with_nulls[1][0], dataset_pairs_with_nulls[2][0])


def test_inject_nulls_should_be_the_same_mar_train_sets_with_nulls(benchmark):
    evaluation_scenarios = ['mar_mar3', 'mar_mnar3']
    experiment_seed = 100
    dataset_pairs_with_nulls = []
    for evaluation_scenario in evaluation_scenarios:
        data_loader = copy.deepcopy(benchmark.init_data_loader)

        # Split and preprocess the dataset
        X_train_val, X_test, y_train_val, y_test = train_test_split(data_loader.X_data,
                                                                    data_loader.y_data,
                                                                    test_size=benchmark.test_set_fraction,
                                                                    random_state=experiment_seed)
        # Inject nulls
        X_train_val_with_nulls, X_test_with_nulls = benchmark._inject_nulls(X_train_val=X_train_val,
                                                                            X_test=X_test,
                                                                            evaluation_scenario=evaluation_scenario,
                                                                            experiment_seed=experiment_seed)

        dataset_pairs_with_nulls.append((X_train_val_with_nulls, X_test_with_nulls))

    assert compare_dfs(dataset_pairs_with_nulls[0][0], dataset_pairs_with_nulls[1][0])
