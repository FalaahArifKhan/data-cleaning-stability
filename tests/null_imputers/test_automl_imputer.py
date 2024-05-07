import pytest
import pathlib

from source.utils.common_helpers import get_injection_scenarios
from source.null_imputers.imputation_methods import impute_with_automl
from configs.constants import ErrorRepairMethod
from configs.datasets_config import ACS_INCOME_DATASET

from tests import assert_nested_dicts_equal


@pytest.fixture(scope="function")
def null_imputer_name():
    return ErrorRepairMethod.automl.value


@pytest.fixture(scope="function")
def automl_kwargs():
    return {"max_trials": 3, "tuner": None, "validation_split": 0.2, "epochs": 3}


# Test if output of automl does not contain nulls
def test_automl_imputer_no_nulls(acs_income_dataset_params, null_imputer_name,
                                 mcar_evaluation_scenario, common_seed, automl_kwargs):
    # Init function variables
    dataset_name = ACS_INCOME_DATASET
    evaluation_scenario = mcar_evaluation_scenario
    experiment_seed = common_seed
    imputation_kwargs = automl_kwargs

    train_injection_scenario, _ = get_injection_scenarios(evaluation_scenario)
    train_injection_strategy = train_injection_scenario[:-1]
    hyperparams = dict()

    (X_train_with_nulls, X_tests_with_nulls_lst,
     train_numerical_null_columns, train_categorical_null_columns,
     numerical_columns, categorical_columns) = acs_income_dataset_params

    # Apply automl
    output_path = (pathlib.Path(__file__).parent.parent.parent
                   .joinpath('results')
                   .joinpath('tests')
                   .joinpath(null_imputer_name)
                   .joinpath(dataset_name)
                   .joinpath(evaluation_scenario)
                   .joinpath(str(experiment_seed)))
    imputation_kwargs.update({'directory': output_path})

    imputation_kwargs.update({'experiment_seed': experiment_seed})
    X_train_imputed, X_tests_imputed_lst, null_imputer_params_dct = (
        impute_with_automl(X_train_with_nulls=X_train_with_nulls,
                           X_tests_with_nulls_lst=X_tests_with_nulls_lst,
                           numeric_columns_with_nulls=train_numerical_null_columns,
                           categorical_columns_with_nulls=train_categorical_null_columns,
                           hyperparams=hyperparams,
                           **imputation_kwargs))

    # Check if there are any nulls in the output
    assert not X_train_imputed.isnull().any().any(), "X_train_imputed contains null values"
    assert not X_tests_imputed_lst[0].isnull().any().any(), "X_tests_imputed_lst[0] contains null values"
    assert not X_tests_imputed_lst[1].isnull().any().any(), "X_tests_imputed_lst[1] contains null values"


# Test if automl returns same results with the same seed
def test_automl_imputer_same_seed(acs_income_dataset_params, null_imputer_name,
                                  mcar_evaluation_scenario, common_seed, automl_kwargs):
    # Init function variables
    dataset_name = ACS_INCOME_DATASET
    evaluation_scenario = mcar_evaluation_scenario
    experiment_seed = common_seed
    imputation_kwargs = automl_kwargs

    train_injection_scenario, _ = get_injection_scenarios(evaluation_scenario)
    train_injection_strategy = train_injection_scenario[:-1]
    hyperparams = dict()

    (X_train_with_nulls, X_tests_with_nulls_lst,
     train_numerical_null_columns, train_categorical_null_columns,
     numerical_columns, categorical_columns) = acs_income_dataset_params

    # Apply automl
    output_path = (pathlib.Path(__file__).parent.parent.parent
                   .joinpath('results')
                   .joinpath('tests')
                   .joinpath(null_imputer_name)
                   .joinpath(dataset_name)
                   .joinpath(evaluation_scenario)
                   .joinpath(str(experiment_seed)))
    imputation_kwargs.update({'directory': output_path})

    imputation_kwargs.update({'experiment_seed': experiment_seed})
    X_train_imputed1, X_test_imputed_lst1, null_imputer_params_dct1 = (
        impute_with_automl(X_train_with_nulls=X_train_with_nulls,
                           X_tests_with_nulls_lst=X_tests_with_nulls_lst,
                           numeric_columns_with_nulls=train_numerical_null_columns,
                           categorical_columns_with_nulls=train_categorical_null_columns,
                           hyperparams=hyperparams,
                           **imputation_kwargs)
    )
    X_train_imputed2, X_test_imputed_lst2, null_imputer_params_dct2 = (
        impute_with_automl(X_train_with_nulls=X_train_with_nulls,
                           X_tests_with_nulls_lst=X_tests_with_nulls_lst,
                           numeric_columns_with_nulls=train_numerical_null_columns,
                           categorical_columns_with_nulls=train_categorical_null_columns,
                           hyperparams=hyperparams,
                           **imputation_kwargs)
    )

    # Check if the results are identical
    assert X_train_imputed1.equals(X_train_imputed2), "X_train_imputed from automl are not identical"
    assert X_test_imputed_lst1[0].equals(X_test_imputed_lst2[0]), "X_test_imputed_lst1[0] from automl are not identical"
    assert X_test_imputed_lst1[1].equals(X_test_imputed_lst2[1]), "X_test_imputed_lst1[1] from automl are not identical"
    assert_nested_dicts_equal(null_imputer_params_dct1, null_imputer_params_dct2,
                              assert_msg="null_imputer_params_dct from automl are not identical")


# Test if automl returns different results for different seeds
def test_automl_imputer_diff_seed(acs_income_dataset_params, null_imputer_name,
                                  mcar_evaluation_scenario, common_seed, automl_kwargs):
    # Init function variables
    dataset_name = ACS_INCOME_DATASET
    evaluation_scenario = mcar_evaluation_scenario
    experiment_seed = common_seed
    imputation_kwargs = automl_kwargs

    train_injection_scenario, _ = get_injection_scenarios(evaluation_scenario)
    train_injection_strategy = train_injection_scenario[:-1]
    hyperparams = dict()

    (X_train_with_nulls, X_tests_with_nulls_lst,
     train_numerical_null_columns, train_categorical_null_columns,
     numerical_columns, categorical_columns) = acs_income_dataset_params

    # Apply automl
    output_path = (pathlib.Path(__file__).parent.parent.parent
                   .joinpath('results')
                   .joinpath('tests')
                   .joinpath(null_imputer_name)
                   .joinpath(dataset_name)
                   .joinpath(evaluation_scenario)
                   .joinpath(str(experiment_seed)))
    imputation_kwargs.update({'directory': output_path})

    imputation_kwargs.update({'experiment_seed': 100})
    X_train_imputed1, X_test_imputed_lst1, null_imputer_params_dct1 = (
        impute_with_automl(X_train_with_nulls=X_train_with_nulls,
                           X_tests_with_nulls_lst=X_tests_with_nulls_lst,
                           numeric_columns_with_nulls=train_numerical_null_columns,
                           categorical_columns_with_nulls=train_categorical_null_columns,
                           hyperparams=hyperparams,
                           **imputation_kwargs)
    )

    imputation_kwargs.update({'experiment_seed': 200})
    X_train_imputed2, X_test_imputed_lst2, null_imputer_params_dct2 = (
        impute_with_automl(X_train_with_nulls=X_train_with_nulls,
                           X_tests_with_nulls_lst=X_tests_with_nulls_lst,
                           numeric_columns_with_nulls=train_numerical_null_columns,
                           categorical_columns_with_nulls=train_categorical_null_columns,
                           hyperparams=hyperparams,
                           **imputation_kwargs)
    )

    # Check if the results are identical
    assert not X_train_imputed1.equals(X_train_imputed2), "X_train_imputed from automl is the same for different seeds"
    assert not X_test_imputed_lst1[0].equals(X_test_imputed_lst2[0]), "X_test_imputed_lst2[0] from automl is the same for different seeds"
    assert not X_test_imputed_lst1[1].equals(X_test_imputed_lst2[1]), "X_test_imputed_lst2[1] from automl is the same for different seeds"
