import copy
import pytest
from sklearn.model_selection import train_test_split

from tests import compare_dfs, get_df_condition
from source.custom_classes.benchmark import Benchmark
from configs.constants import ACS_INCOME_DATASET, ErrorRepairMethod, MLModels, EXPERIMENT_RUN_SEEDS
from configs.evaluation_scenarios_config import EVALUATION_SCENARIOS_CONFIG


@pytest.fixture(scope='module')
def folk_benchmark():
    benchmark = Benchmark(dataset_name=ACS_INCOME_DATASET,
                          null_imputers=[ErrorRepairMethod.median_mode.value],
                          model_names=[MLModels.lr_clf.value])

    return benchmark

# ====================================================================
# Test error injection
# ====================================================================
def test_inject_nulls_should_be_the_same_mcar_train_sets_with_nulls(folk_benchmark):
    evaluation_scenarios = ['mcar_mcar2', 'mcar_mar2', 'mcar_mnar2']
    experiment_seed = 100
    dataset_pairs_with_nulls = []
    for evaluation_scenario in evaluation_scenarios:
        data_loader = copy.deepcopy(folk_benchmark.init_data_loader)

        # Split and preprocess the dataset
        X_train_val, X_test, y_train_val, y_test = train_test_split(data_loader.X_data,
                                                                    data_loader.y_data,
                                                                    test_size=folk_benchmark.test_set_fraction,
                                                                    random_state=experiment_seed)
        # Inject nulls
        X_train_val_with_nulls, X_test_with_nulls = folk_benchmark._inject_nulls(X_train_val=X_train_val,
                                                                                 X_test=X_test,
                                                                                 evaluation_scenario=evaluation_scenario,
                                                                                 experiment_seed=experiment_seed)

        dataset_pairs_with_nulls.append((X_train_val_with_nulls, X_test_with_nulls))

    assert compare_dfs(dataset_pairs_with_nulls[0][0], dataset_pairs_with_nulls[1][0])

    assert compare_dfs(dataset_pairs_with_nulls[1][0], dataset_pairs_with_nulls[2][0])


def test_inject_nulls_should_be_the_same_mar_train_sets_with_nulls(folk_benchmark):
    evaluation_scenarios = ['mar_mar3', 'mar_mnar3']
    experiment_seed = 100
    dataset_pairs_with_nulls = []
    for evaluation_scenario in evaluation_scenarios:
        data_loader = copy.deepcopy(folk_benchmark.init_data_loader)

        # Split and preprocess the dataset
        X_train_val, X_test, y_train_val, y_test = train_test_split(data_loader.X_data,
                                                                    data_loader.y_data,
                                                                    test_size=folk_benchmark.test_set_fraction,
                                                                    random_state=experiment_seed)
        # Inject nulls
        X_train_val_with_nulls, X_test_with_nulls = folk_benchmark._inject_nulls(X_train_val=X_train_val,
                                                                                 X_test=X_test,
                                                                                 evaluation_scenario=evaluation_scenario,
                                                                                 experiment_seed=experiment_seed)

        dataset_pairs_with_nulls.append((X_train_val_with_nulls, X_test_with_nulls))

    assert compare_dfs(dataset_pairs_with_nulls[0][0], dataset_pairs_with_nulls[1][0])


def test_inject_nulls_into_one_set_should_apply_mnar_scenario_for_multiple_columns(folk_benchmark):
    experiment_seed = 200
    injection_strategy = 'MNAR'
    error_rate_idx = 2

    X_train_val, X_test, _, _ = train_test_split(folk_benchmark.init_data_loader.X_data,
                                                 folk_benchmark.init_data_loader.y_data,
                                                 test_size=folk_benchmark.test_set_fraction,
                                                 random_state=experiment_seed)
    X_test_with_nulls = folk_benchmark._inject_nulls_into_one_set(df=X_test,
                                                                  injection_strategy=injection_strategy,
                                                                  error_rate_idx=error_rate_idx,
                                                                  experiment_seed=experiment_seed)

    for mnar_injection_scenario in EVALUATION_SCENARIOS_CONFIG[folk_benchmark.dataset_name][injection_strategy]:
        missing_feature = mnar_injection_scenario['missing_features'][0]
        error_rate = mnar_injection_scenario['setting']['error_rates'][error_rate_idx]
        condition_value = mnar_injection_scenario['setting']['condition'][1]

        actual_column_nulls_count = X_test_with_nulls[missing_feature].isnull().sum()
        df_condition = get_df_condition(df=X_test,
                                        condition_col=missing_feature,
                                        condition_val=condition_value,
                                        include_val=True)
        assert actual_column_nulls_count == int(X_test[df_condition].shape[0] * error_rate)
