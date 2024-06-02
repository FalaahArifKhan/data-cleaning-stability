import copy
import pytest
import pathlib
from unittest import mock
from unittest.mock import MagicMock
from sklearn.model_selection import train_test_split

from tests import compare_dfs, get_df_condition, compare_base_flow_datasets
from source.custom_classes.benchmark import Benchmark
from source.utils.common_helpers import get_injection_scenarios
from configs.constants import ACS_INCOME_DATASET, LAW_SCHOOL_DATASET, ErrorRepairMethod, MLModels, ErrorInjectionStrategy
from configs.scenarios_config import ERROR_INJECTION_SCENARIOS_CONFIG


@pytest.fixture(scope='function')
def folk_benchmark():
    benchmark = Benchmark(dataset_name=ACS_INCOME_DATASET,
                          null_imputers=[ErrorRepairMethod.median_mode.value],
                          model_names=[MLModels.lr_clf.value])

    return benchmark

# ====================================================================
# Test error injection
# ====================================================================
def test_inject_nulls_should_be_the_same_mcar_train_sets_with_nulls(folk_benchmark):
    evaluation_scenarios = ['exp1_mcar3', 'exp1_mcar3']
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
        X_train_val_with_nulls, X_tests_with_nulls_lst = folk_benchmark._inject_nulls(X_train_val=X_train_val,
                                                                                      X_test=X_test,
                                                                                      evaluation_scenario=evaluation_scenario,
                                                                                      experiment_seed=experiment_seed)

        dataset_pairs_with_nulls.append((X_train_val_with_nulls, X_tests_with_nulls_lst))

    assert compare_dfs(dataset_pairs_with_nulls[0][0], dataset_pairs_with_nulls[1][0])
    assert compare_dfs(dataset_pairs_with_nulls[0][1][0], dataset_pairs_with_nulls[1][1][0])
    assert compare_dfs(dataset_pairs_with_nulls[0][1][1], dataset_pairs_with_nulls[1][1][1])


def test_inject_nulls_should_be_the_same_mnar_train_sets_with_nulls(folk_benchmark):
    evaluation_scenarios = ['exp1_mnar3', 'exp1_mnar3']
    experiment_seed = 200
    dataset_pairs_with_nulls = []
    for evaluation_scenario in evaluation_scenarios:
        data_loader = copy.deepcopy(folk_benchmark.init_data_loader)

        # Split and preprocess the dataset
        X_train_val, X_test, y_train_val, y_test = train_test_split(data_loader.X_data,
                                                                    data_loader.y_data,
                                                                    test_size=folk_benchmark.test_set_fraction,
                                                                    random_state=experiment_seed)
        # Inject nulls
        X_train_val_with_nulls, X_tests_with_nulls_lst = folk_benchmark._inject_nulls(X_train_val=X_train_val,
                                                                                      X_test=X_test,
                                                                                      evaluation_scenario=evaluation_scenario,
                                                                                      experiment_seed=experiment_seed)

        dataset_pairs_with_nulls.append((X_train_val_with_nulls, X_tests_with_nulls_lst))

    assert compare_dfs(dataset_pairs_with_nulls[0][0], dataset_pairs_with_nulls[1][0])
    assert compare_dfs(dataset_pairs_with_nulls[0][1][0], dataset_pairs_with_nulls[1][1][0])


def test_inject_nulls_into_one_set_for_mcar_evaluation_scenario(folk_benchmark):
    experiment_seed = 300
    evaluation_scenario = 'exp1_mcar3'
    train_injection_scenario, _ = get_injection_scenarios(evaluation_scenario)

    X_train_val, X_test, _, _ = train_test_split(folk_benchmark.init_data_loader.X_data,
                                                 folk_benchmark.init_data_loader.y_data,
                                                 test_size=folk_benchmark.test_set_fraction,
                                                 random_state=experiment_seed)
    X_train_val_with_nulls = folk_benchmark._inject_nulls_into_one_set(df=X_train_val,
                                                                       injection_scenario=train_injection_scenario,
                                                                       experiment_seed=experiment_seed)

    injection_strategy, error_rate_str = train_injection_scenario[:-1], train_injection_scenario[-1]
    error_rate_idx = int(error_rate_str) - 1
    scenario_for_dataset = ERROR_INJECTION_SCENARIOS_CONFIG[folk_benchmark.dataset_name][injection_strategy][0]
    error_rate = scenario_for_dataset['setting']['error_rates'][error_rate_idx]
    missing_features = scenario_for_dataset['missing_features']

    actual_column_nulls_count = X_train_val_with_nulls[missing_features].isnull().sum().sum()
    assert actual_column_nulls_count == int(X_train_val.shape[0] * error_rate)


def test_inject_nulls_into_one_set_for_mar_evaluation_scenario(folk_benchmark):
    experiment_seed = 400
    test_injection_scenario = 'MAR1'

    X_train_val, X_test, _, _ = train_test_split(folk_benchmark.init_data_loader.X_data,
                                                 folk_benchmark.init_data_loader.y_data,
                                                 test_size=folk_benchmark.test_set_fraction,
                                                 random_state=experiment_seed)
    X_test_with_nulls = folk_benchmark._inject_nulls_into_one_set(df=X_test,
                                                                  injection_scenario=test_injection_scenario,
                                                                  experiment_seed=experiment_seed)

    injection_strategy, error_rate_str = test_injection_scenario[:-1], test_injection_scenario[-1]
    error_rate_idx = int(error_rate_str) - 1
    for injection_scenario in ERROR_INJECTION_SCENARIOS_CONFIG[folk_benchmark.dataset_name][injection_strategy]:
        missing_features = injection_scenario['missing_features']
        error_rate = injection_scenario['setting']['error_rates'][error_rate_idx]
        condition_column, condition_value = injection_scenario['setting']['condition']

        df_condition = get_df_condition(df=X_test,
                                        condition_col=condition_column,
                                        condition_val=condition_value,
                                        include_val=True)
        actual_column_nulls_count = X_test_with_nulls[df_condition][missing_features].isnull().sum().sum()
        assert actual_column_nulls_count == int(X_test[df_condition].shape[0] * error_rate)


def test_inject_nulls_into_one_set_should_apply_mnar_scenario_for_multiple_columns(folk_benchmark):
    experiment_seed = 200
    test_injection_scenario = 'MNAR3'

    X_train_val, X_test, _, _ = train_test_split(folk_benchmark.init_data_loader.X_data,
                                                 folk_benchmark.init_data_loader.y_data,
                                                 test_size=folk_benchmark.test_set_fraction,
                                                 random_state=experiment_seed)
    X_test_with_nulls = folk_benchmark._inject_nulls_into_one_set(df=X_test,
                                                                  injection_scenario=test_injection_scenario,
                                                                  experiment_seed=experiment_seed)

    injection_strategy, error_rate_str = test_injection_scenario[:-1], test_injection_scenario[-1]
    error_rate_idx = int(error_rate_str) - 1
    for injection_scenario in ERROR_INJECTION_SCENARIOS_CONFIG[folk_benchmark.dataset_name][injection_strategy]:
        missing_feature = injection_scenario['missing_features'][0]
        error_rate = injection_scenario['setting']['error_rates'][error_rate_idx]
        condition_value = injection_scenario['setting']['condition'][1]

        df_condition = get_df_condition(df=X_test,
                                        condition_col=missing_feature,
                                        condition_val=condition_value,
                                        include_val=True)
        actual_column_nulls_count = X_test_with_nulls[df_condition][missing_feature].isnull().sum()
        assert actual_column_nulls_count == int(X_test[df_condition].shape[0] * error_rate)


def test_inject_nulls_into_one_set_for_mixed_evaluation_scenario(folk_benchmark):
    experiment_seed = 400
    test_injection_scenario = 'MCAR1 & MAR1 & MNAR1'

    X_train_val, X_test, _, _ = train_test_split(folk_benchmark.init_data_loader.X_data,
                                                 folk_benchmark.init_data_loader.y_data,
                                                 test_size=folk_benchmark.test_set_fraction,
                                                 random_state=experiment_seed)
    X_test_with_nulls = folk_benchmark._inject_nulls_into_one_set(df=X_test,
                                                                  injection_scenario=test_injection_scenario,
                                                                  experiment_seed=experiment_seed)

    single_injection_scenarios = [s.strip() for s in test_injection_scenario.split('&')]
    for single_injection_scenario in single_injection_scenarios:
        injection_strategy, error_rate_str = single_injection_scenario[:-1], single_injection_scenario[-1]
        error_rate_idx = int(error_rate_str) - 1

        if injection_strategy.upper() == 'MCAR':
            scenario_for_dataset = ERROR_INJECTION_SCENARIOS_CONFIG[folk_benchmark.dataset_name][injection_strategy][0]
            error_rate = scenario_for_dataset['setting']['error_rates'][error_rate_idx]
            missing_features = scenario_for_dataset['missing_features']
            actual_column_nulls_count = X_test_with_nulls[missing_features].isnull().sum().sum()
            assert actual_column_nulls_count >= int(X_test.shape[0] * error_rate)

        elif injection_strategy.upper() == 'MAR':
            for injection_scenario in ERROR_INJECTION_SCENARIOS_CONFIG[folk_benchmark.dataset_name][injection_strategy]:
                missing_features = injection_scenario['missing_features']
                error_rate = injection_scenario['setting']['error_rates'][error_rate_idx]
                condition_column, condition_value = injection_scenario['setting']['condition']

                df_condition = get_df_condition(df=X_test,
                                                condition_col=condition_column,
                                                condition_val=condition_value,
                                                include_val=True)
                actual_column_nulls_count = X_test_with_nulls[df_condition][missing_features].isnull().sum().sum()
                assert actual_column_nulls_count >= int(X_test[df_condition].shape[0] * error_rate)

        else:
            for injection_scenario in ERROR_INJECTION_SCENARIOS_CONFIG[folk_benchmark.dataset_name][injection_strategy]:
                missing_feature = injection_scenario['missing_features'][0]
                error_rate = injection_scenario['setting']['error_rates'][error_rate_idx]
                condition_value = injection_scenario['setting']['condition'][1]

                df_condition = get_df_condition(df=X_test,
                                                condition_col=missing_feature,
                                                condition_val=condition_value,
                                                include_val=True)
                actual_column_nulls_count = X_test_with_nulls[df_condition][missing_feature].isnull().sum()
                assert actual_column_nulls_count >= int(X_test[df_condition].shape[0] * error_rate)


# ====================================================================
# Test sequence of test sets with nulls
# ====================================================================
def test_inject_nulls_should_preserve_mcar_scenario_test_sets_sequence(folk_benchmark):
    evaluation_scenario = 'exp1_mcar3'
    experiment_seed = 100
    data_loader = folk_benchmark.init_data_loader

    # Split and preprocess the dataset
    X_train_val, X_test, y_train_val, y_test = train_test_split(data_loader.X_data,
                                                                data_loader.y_data,
                                                                test_size=folk_benchmark.test_set_fraction,
                                                                random_state=experiment_seed)
    # Inject nulls
    X_train_val_with_nulls, X_tests_with_nulls_lst = folk_benchmark._inject_nulls(X_train_val=X_train_val,
                                                                                  X_test=X_test,
                                                                                  evaluation_scenario=evaluation_scenario,
                                                                                  experiment_seed=experiment_seed)

    expected_test_injection_scenarios = ['MCAR3', 'MAR3', 'MNAR3']
    for test_set_idx, injection_scenario in enumerate(expected_test_injection_scenarios):
        X_test_with_nulls = X_tests_with_nulls_lst[test_set_idx]
        injection_strategy, error_rate_str = injection_scenario[:-1], injection_scenario[-1]
        error_rate_idx = int(error_rate_str) - 1

        for injection_scenario_config in ERROR_INJECTION_SCENARIOS_CONFIG[folk_benchmark.dataset_name][injection_strategy]:
            missing_features = injection_scenario_config['missing_features']
            error_rate = injection_scenario_config['setting']['error_rates'][error_rate_idx]

            if injection_strategy == ErrorInjectionStrategy.mcar.value:
                actual_column_nulls_count = X_test_with_nulls[missing_features].isnull().sum().sum()
                assert actual_column_nulls_count == int(X_test.shape[0] * error_rate)
            else:
                condition_column, condition_value = injection_scenario_config['setting']['condition']
                df_condition = get_df_condition(df=X_test,
                                                condition_col=condition_column,
                                                condition_val=condition_value,
                                                include_val=True)
                actual_column_nulls_count = X_test_with_nulls[df_condition][missing_features].isnull().sum().sum()
                assert actual_column_nulls_count == int(X_test[df_condition].shape[0] * error_rate)


def test_inject_nulls_should_preserve_mar_scenario_test_sets_sequence(folk_benchmark):
    evaluation_scenario = 'exp2_3_mar5'
    experiment_seed = 200
    data_loader = folk_benchmark.init_data_loader

    # Split and preprocess the dataset
    X_train_val, X_test, y_train_val, y_test = train_test_split(data_loader.X_data,
                                                                data_loader.y_data,
                                                                test_size=folk_benchmark.test_set_fraction,
                                                                random_state=experiment_seed)
    # Inject nulls
    X_train_val_with_nulls, X_tests_with_nulls_lst = folk_benchmark._inject_nulls(X_train_val=X_train_val,
                                                                                  X_test=X_test,
                                                                                  evaluation_scenario=evaluation_scenario,
                                                                                  experiment_seed=experiment_seed)

    expected_test_injection_scenarios = ['MCAR3', 'MAR3', 'MNAR3']
    for test_set_idx, injection_scenario in enumerate(expected_test_injection_scenarios):
        X_test_with_nulls = X_tests_with_nulls_lst[test_set_idx]
        injection_strategy, error_rate_str = injection_scenario[:-1], injection_scenario[-1]
        error_rate_idx = int(error_rate_str) - 1

        for injection_scenario_config in ERROR_INJECTION_SCENARIOS_CONFIG[folk_benchmark.dataset_name][injection_strategy]:
            missing_features = injection_scenario_config['missing_features']
            error_rate = injection_scenario_config['setting']['error_rates'][error_rate_idx]

            if injection_strategy == ErrorInjectionStrategy.mcar.value:
                actual_column_nulls_count = X_test_with_nulls[missing_features].isnull().sum().sum()
                assert actual_column_nulls_count == int(X_test.shape[0] * error_rate)
            else:
                condition_column, condition_value = injection_scenario_config['setting']['condition']
                df_condition = get_df_condition(df=X_test,
                                                condition_col=condition_column,
                                                condition_val=condition_value,
                                                include_val=True)
                actual_column_nulls_count = X_test_with_nulls[df_condition][missing_features].isnull().sum().sum()
                assert actual_column_nulls_count == int(X_test[df_condition].shape[0] * error_rate)


def test_inject_nulls_should_preserve_mnar_scenario_test_sets_sequence(folk_benchmark):
    evaluation_scenario = 'exp2_3_mnar3'
    experiment_seed = 300
    data_loader = folk_benchmark.init_data_loader

    # Split and preprocess the dataset
    X_train_val, X_test, y_train_val, y_test = train_test_split(data_loader.X_data,
                                                                data_loader.y_data,
                                                                test_size=folk_benchmark.test_set_fraction,
                                                                random_state=experiment_seed)
    # Inject nulls
    X_train_val_with_nulls, X_tests_with_nulls_lst = folk_benchmark._inject_nulls(X_train_val=X_train_val,
                                                                                  X_test=X_test,
                                                                                  evaluation_scenario=evaluation_scenario,
                                                                                  experiment_seed=experiment_seed)

    expected_test_injection_scenarios = [
        'MCAR1', 'MAR1', 'MNAR1',
        'MCAR2', 'MAR2', 'MNAR2',
        'MCAR3', 'MAR3', 'MNAR3',
        'MCAR4', 'MAR4', 'MNAR4',
        'MCAR5', 'MAR5', 'MNAR5',
    ]
    for test_set_idx, injection_scenario in enumerate(expected_test_injection_scenarios):
        X_test_with_nulls = X_tests_with_nulls_lst[test_set_idx]
        injection_strategy, error_rate_str = injection_scenario[:-1], injection_scenario[-1]
        error_rate_idx = int(error_rate_str) - 1

        for injection_scenario_config in ERROR_INJECTION_SCENARIOS_CONFIG[folk_benchmark.dataset_name][injection_strategy]:
            missing_features = injection_scenario_config['missing_features']
            error_rate = injection_scenario_config['setting']['error_rates'][error_rate_idx]

            if injection_strategy == ErrorInjectionStrategy.mcar.value:
                actual_column_nulls_count = X_test_with_nulls[missing_features].isnull().sum().sum()
                assert actual_column_nulls_count == int(X_test.shape[0] * error_rate)
            else:
                condition_column, condition_value = injection_scenario_config['setting']['condition']
                df_condition = get_df_condition(df=X_test,
                                                condition_col=condition_column,
                                                condition_val=condition_value,
                                                include_val=True)
                actual_column_nulls_count = X_test_with_nulls[df_condition][missing_features].isnull().sum().sum()
                assert actual_column_nulls_count == int(X_test[df_condition].shape[0] * error_rate)


# ====================================================================
# Test load_imputed_train_test_sets
# ====================================================================
@mock.patch.multiple(Benchmark,
                     _save_imputation_metrics_to_db=MagicMock())
def test_load_imputed_train_test_sets_for_median_mode_and_law_school_mcar3():
    dataset_name = LAW_SCHOOL_DATASET
    null_imputer_name = ErrorRepairMethod.median_mode.value
    evaluation_scenario = 'exp1_mcar3'
    experiment_seed = 100
    tune_imputers = True
    save_imputed_datasets = False

    save_sets_dir_path = (pathlib.Path(__file__).parent.parent
                          .joinpath('files_for_tests')
                          .joinpath('results')
                          .joinpath('imputed_datasets')
                          .joinpath(dataset_name)
                          .joinpath(null_imputer_name)
                          .joinpath(evaluation_scenario)
                          .joinpath(str(experiment_seed)))
    benchmark = Benchmark(dataset_name=dataset_name,
                          null_imputers=[null_imputer_name],
                          model_names=[])

    # Create a mock for save_sets_dir_path in benchmark.load_imputed_train_test_sets()
    with mock.patch('source.custom_classes.benchmark.pathlib.Path') as mock_path:
        # Setup the mock to return a specific path when joined
        instance = mock_path.return_value  # this is the instance returned when Path() is called
        instance.parent.parent.parent.joinpath.return_value.joinpath.return_value.joinpath.return_value.joinpath.return_value.joinpath.return_value.joinpath.return_value = save_sets_dir_path

        expected_main_base_flow_dataset, expected_extra_base_flow_datasets = (
            benchmark.inject_and_impute_nulls(data_loader=copy.deepcopy(benchmark.init_data_loader),
                                              null_imputer_name=null_imputer_name,
                                              evaluation_scenario=evaluation_scenario,
                                              tune_imputers=tune_imputers,
                                              experiment_seed=experiment_seed,
                                              save_imputed_datasets=save_imputed_datasets))

        actual_main_base_flow_dataset, actual_extra_base_flow_datasets = (
            benchmark.load_imputed_train_test_sets(data_loader=copy.deepcopy(benchmark.init_data_loader),
                                                   null_imputer_name=null_imputer_name,
                                                   evaluation_scenario=evaluation_scenario,
                                                   experiment_seed=experiment_seed))

        compare_base_flow_datasets(expected_main_base_flow_dataset, actual_main_base_flow_dataset)

        for idx in range(len(expected_extra_base_flow_datasets)):
            compare_base_flow_datasets(expected_extra_base_flow_datasets[idx], actual_extra_base_flow_datasets[idx])


@mock.patch.multiple(Benchmark,
                     _save_imputation_metrics_to_db=MagicMock())
def test_load_imputed_train_test_sets_for_median_mode_and_law_school_mar3():
    dataset_name = LAW_SCHOOL_DATASET
    null_imputer_name = ErrorRepairMethod.median_mode.value
    evaluation_scenario = 'exp1_mar3'
    experiment_seed = 100
    tune_imputers = True
    save_imputed_datasets = False

    save_sets_dir_path = (pathlib.Path(__file__).parent.parent
                          .joinpath('files_for_tests')
                          .joinpath('results')
                          .joinpath('imputed_datasets')
                          .joinpath(dataset_name)
                          .joinpath(null_imputer_name)
                          .joinpath(evaluation_scenario)
                          .joinpath(str(experiment_seed)))
    benchmark = Benchmark(dataset_name=dataset_name,
                          null_imputers=[null_imputer_name],
                          model_names=[])

    # Create a mock for save_sets_dir_path in benchmark.load_imputed_train_test_sets()
    with mock.patch('source.custom_classes.benchmark.pathlib.Path') as mock_path:
        # Setup the mock to return a specific path when joined
        instance = mock_path.return_value  # this is the instance returned when Path() is called
        instance.parent.parent.parent.joinpath.return_value.joinpath.return_value.joinpath.return_value.joinpath.return_value.joinpath.return_value.joinpath.return_value = save_sets_dir_path

        expected_main_base_flow_dataset, expected_extra_base_flow_datasets = (
            benchmark.inject_and_impute_nulls(data_loader=copy.deepcopy(benchmark.init_data_loader),
                                              null_imputer_name=null_imputer_name,
                                              evaluation_scenario=evaluation_scenario,
                                              tune_imputers=tune_imputers,
                                              experiment_seed=experiment_seed,
                                              save_imputed_datasets=save_imputed_datasets))

        actual_main_base_flow_dataset, actual_extra_base_flow_datasets = (
            benchmark.load_imputed_train_test_sets(data_loader=copy.deepcopy(benchmark.init_data_loader),
                                                   null_imputer_name=null_imputer_name,
                                                   evaluation_scenario=evaluation_scenario,
                                                   experiment_seed=experiment_seed))

        compare_base_flow_datasets(expected_main_base_flow_dataset, actual_main_base_flow_dataset)

        for idx in range(len(expected_extra_base_flow_datasets)):
            compare_base_flow_datasets(expected_extra_base_flow_datasets[idx], actual_extra_base_flow_datasets[idx])


@mock.patch.multiple(Benchmark,
                     _save_imputation_metrics_to_db=MagicMock())
def test_load_imputed_train_test_sets_for_median_mode_and_law_school_mnar3():
    dataset_name = LAW_SCHOOL_DATASET
    null_imputer_name = ErrorRepairMethod.median_mode.value
    evaluation_scenario = 'exp1_mnar3'
    experiment_seed = 100
    tune_imputers = True
    save_imputed_datasets = False

    save_sets_dir_path = (pathlib.Path(__file__).parent.parent
                          .joinpath('files_for_tests')
                          .joinpath('results')
                          .joinpath('imputed_datasets')
                          .joinpath(dataset_name)
                          .joinpath(null_imputer_name)
                          .joinpath(evaluation_scenario)
                          .joinpath(str(experiment_seed)))
    benchmark = Benchmark(dataset_name=dataset_name,
                          null_imputers=[null_imputer_name],
                          model_names=[])

    # Create a mock for save_sets_dir_path in benchmark.load_imputed_train_test_sets()
    with mock.patch('source.custom_classes.benchmark.pathlib.Path') as mock_path:
        # Setup the mock to return a specific path when joined
        instance = mock_path.return_value  # this is the instance returned when Path() is called
        instance.parent.parent.parent.joinpath.return_value.joinpath.return_value.joinpath.return_value.joinpath.return_value.joinpath.return_value.joinpath.return_value = save_sets_dir_path

        expected_main_base_flow_dataset, expected_extra_base_flow_datasets = (
            benchmark.inject_and_impute_nulls(data_loader=copy.deepcopy(benchmark.init_data_loader),
                                              null_imputer_name=null_imputer_name,
                                              evaluation_scenario=evaluation_scenario,
                                              tune_imputers=tune_imputers,
                                              experiment_seed=experiment_seed,
                                              save_imputed_datasets=save_imputed_datasets))

        actual_main_base_flow_dataset, actual_extra_base_flow_datasets = (
            benchmark.load_imputed_train_test_sets(data_loader=copy.deepcopy(benchmark.init_data_loader),
                                                   null_imputer_name=null_imputer_name,
                                                   evaluation_scenario=evaluation_scenario,
                                                   experiment_seed=experiment_seed))

        compare_base_flow_datasets(expected_main_base_flow_dataset, actual_main_base_flow_dataset)

        for idx in range(len(expected_extra_base_flow_datasets)):
            compare_base_flow_datasets(expected_extra_base_flow_datasets[idx], actual_extra_base_flow_datasets[idx])


@mock.patch.multiple(Benchmark,
                     _save_imputation_metrics_to_db=MagicMock())
def test_load_imputed_train_test_sets_for_median_mode_and_acs_income_mcar3():
    dataset_name = ACS_INCOME_DATASET
    null_imputer_name = ErrorRepairMethod.median_mode.value
    evaluation_scenario = 'exp1_mcar3'
    experiment_seed = 500
    tune_imputers = True
    save_imputed_datasets = False

    save_sets_dir_path = (pathlib.Path(__file__).parent.parent
                          .joinpath('files_for_tests')
                          .joinpath('results')
                          .joinpath('imputed_datasets')
                          .joinpath(dataset_name)
                          .joinpath(null_imputer_name)
                          .joinpath(evaluation_scenario)
                          .joinpath(str(experiment_seed)))
    benchmark = Benchmark(dataset_name=dataset_name,
                          null_imputers=[null_imputer_name],
                          model_names=[])

    # Create a mock for save_sets_dir_path in benchmark.load_imputed_train_test_sets()
    with mock.patch('source.custom_classes.benchmark.pathlib.Path') as mock_path:
        # Setup the mock to return a specific path when joined
        instance = mock_path.return_value  # this is the instance returned when Path() is called
        instance.parent.parent.parent.joinpath.return_value.joinpath.return_value.joinpath.return_value.joinpath.return_value.joinpath.return_value.joinpath.return_value = save_sets_dir_path

        expected_main_base_flow_dataset, expected_extra_base_flow_datasets = (
            benchmark.inject_and_impute_nulls(data_loader=copy.deepcopy(benchmark.init_data_loader),
                                              null_imputer_name=null_imputer_name,
                                              evaluation_scenario=evaluation_scenario,
                                              tune_imputers=tune_imputers,
                                              experiment_seed=experiment_seed,
                                              save_imputed_datasets=save_imputed_datasets))

        actual_main_base_flow_dataset, actual_extra_base_flow_datasets = (
            benchmark.load_imputed_train_test_sets(data_loader=copy.deepcopy(benchmark.init_data_loader),
                                                   null_imputer_name=null_imputer_name,
                                                   evaluation_scenario=evaluation_scenario,
                                                   experiment_seed=experiment_seed))

        compare_base_flow_datasets(expected_main_base_flow_dataset, actual_main_base_flow_dataset)

        for idx in range(len(expected_extra_base_flow_datasets)):
            compare_base_flow_datasets(expected_extra_base_flow_datasets[idx], actual_extra_base_flow_datasets[idx])


@mock.patch.multiple(Benchmark,
                     _save_imputation_metrics_to_db=MagicMock())
def test_load_imputed_train_test_sets_for_median_mode_and_acs_income_mar3():
    dataset_name = ACS_INCOME_DATASET
    null_imputer_name = ErrorRepairMethod.median_mode.value
    evaluation_scenario = 'exp1_mar3'
    experiment_seed = 500
    tune_imputers = True
    save_imputed_datasets = False

    save_sets_dir_path = (pathlib.Path(__file__).parent.parent
                          .joinpath('files_for_tests')
                          .joinpath('results')
                          .joinpath('imputed_datasets')
                          .joinpath(dataset_name)
                          .joinpath(null_imputer_name)
                          .joinpath(evaluation_scenario)
                          .joinpath(str(experiment_seed)))
    benchmark = Benchmark(dataset_name=dataset_name,
                          null_imputers=[null_imputer_name],
                          model_names=[])

    # Create a mock for save_sets_dir_path in benchmark.load_imputed_train_test_sets()
    with mock.patch('source.custom_classes.benchmark.pathlib.Path') as mock_path:
        # Setup the mock to return a specific path when joined
        instance = mock_path.return_value  # this is the instance returned when Path() is called
        instance.parent.parent.parent.joinpath.return_value.joinpath.return_value.joinpath.return_value.joinpath.return_value.joinpath.return_value.joinpath.return_value = save_sets_dir_path

        expected_main_base_flow_dataset, expected_extra_base_flow_datasets = (
            benchmark.inject_and_impute_nulls(data_loader=copy.deepcopy(benchmark.init_data_loader),
                                              null_imputer_name=null_imputer_name,
                                              evaluation_scenario=evaluation_scenario,
                                              tune_imputers=tune_imputers,
                                              experiment_seed=experiment_seed,
                                              save_imputed_datasets=save_imputed_datasets))

        actual_main_base_flow_dataset, actual_extra_base_flow_datasets = (
            benchmark.load_imputed_train_test_sets(data_loader=copy.deepcopy(benchmark.init_data_loader),
                                                   null_imputer_name=null_imputer_name,
                                                   evaluation_scenario=evaluation_scenario,
                                                   experiment_seed=experiment_seed))

        compare_base_flow_datasets(expected_main_base_flow_dataset, actual_main_base_flow_dataset)

        for idx in range(len(expected_extra_base_flow_datasets)):
            compare_base_flow_datasets(expected_extra_base_flow_datasets[idx], actual_extra_base_flow_datasets[idx])


@mock.patch.multiple(Benchmark,
                     _save_imputation_metrics_to_db=MagicMock())
def test_load_imputed_train_test_sets_for_median_mode_and_acs_income_mnar3():
    dataset_name = ACS_INCOME_DATASET
    null_imputer_name = ErrorRepairMethod.median_mode.value
    evaluation_scenario = 'exp1_mnar3'
    experiment_seed = 500
    tune_imputers = True
    save_imputed_datasets = False

    save_sets_dir_path = (pathlib.Path(__file__).parent.parent
                              .joinpath('files_for_tests')
                              .joinpath('results')
                              .joinpath('imputed_datasets')
                              .joinpath(dataset_name)
                              .joinpath(null_imputer_name)
                              .joinpath(evaluation_scenario)
                              .joinpath(str(experiment_seed)))
    benchmark = Benchmark(dataset_name=dataset_name,
                          null_imputers=[null_imputer_name],
                          model_names=[])

    # Create a mock for save_sets_dir_path in benchmark.load_imputed_train_test_sets()
    with mock.patch('source.custom_classes.benchmark.pathlib.Path') as mock_path:

        # Setup the mock to return a specific path when joined
        instance = mock_path.return_value  # this is the instance returned when Path() is called
        instance.parent.parent.parent.joinpath.return_value.joinpath.return_value.joinpath.return_value.joinpath.return_value.joinpath.return_value.joinpath.return_value = save_sets_dir_path

        expected_main_base_flow_dataset, expected_extra_base_flow_datasets = (
            benchmark.inject_and_impute_nulls(data_loader=copy.deepcopy(benchmark.init_data_loader),
                                              null_imputer_name=null_imputer_name,
                                              evaluation_scenario=evaluation_scenario,
                                              tune_imputers=tune_imputers,
                                              experiment_seed=experiment_seed,
                                              save_imputed_datasets=save_imputed_datasets))

        actual_main_base_flow_dataset, actual_extra_base_flow_datasets = (
            benchmark.load_imputed_train_test_sets(data_loader=copy.deepcopy(benchmark.init_data_loader),
                                                   null_imputer_name=null_imputer_name,
                                                   evaluation_scenario=evaluation_scenario,
                                                   experiment_seed=experiment_seed))

        compare_base_flow_datasets(expected_main_base_flow_dataset, actual_main_base_flow_dataset)

        for idx in range(len(expected_extra_base_flow_datasets)):
            compare_base_flow_datasets(expected_extra_base_flow_datasets[idx], actual_extra_base_flow_datasets[idx])
