import pytest
import pathlib

import source.null_imputers.datawig_imputer as datawig_imputer
from source.validation import parse_evaluation_scenario
from configs.datasets_config import ACS_INCOME_DATASET
from configs.constants import ErrorRepairMethod
from configs.null_imputers_config import NULL_IMPUTERS_HYPERPARAMS

from tests import assert_nested_dicts_equal


@pytest.fixture(scope="function")
def null_imputer_name():
    return ErrorRepairMethod.datawig.value


@pytest.fixture(scope="function")
def datawig_kwargs():
    return {"precision_threshold": 0.0, "num_epochs": 3, "iterations": 1}


# Test if output of datawig does not contain nulls
def test_datawig_imputer_no_nulls(acs_income_dataset_params, null_imputer_name,
                                  mcar_evaluation_scenario, common_seed, datawig_kwargs):
    # Init function variables
    dataset_name = ACS_INCOME_DATASET
    evaluation_scenario = mcar_evaluation_scenario
    experiment_seed = common_seed
    imputation_kwargs = datawig_kwargs

    train_injection_strategy, _ = parse_evaluation_scenario(evaluation_scenario)
    hyperparams = NULL_IMPUTERS_HYPERPARAMS.get(null_imputer_name, {}).get(dataset_name, {}).get(train_injection_strategy, {})

    (X_train_with_nulls, X_test_with_nulls,
     train_numerical_null_columns, train_categorical_null_columns,
     numerical_columns, categorical_columns) = acs_income_dataset_params

    # Apply datawig
    output_path = (pathlib.Path(__file__).parent.parent.parent.joinpath('results')
                       .joinpath("tests")
                       .joinpath(null_imputer_name)
                       .joinpath(dataset_name)
                       .joinpath(evaluation_scenario)
                       .joinpath(str(experiment_seed)))

    imputation_kwargs.update({'experiment_seed': experiment_seed})
    X_train_imputed, X_test_imputed, null_imputer_params_dct = (
        datawig_imputer.complete(X_train_with_nulls=X_train_with_nulls,
                                 X_test_with_nulls=X_test_with_nulls,
                                 numeric_columns_with_nulls=train_numerical_null_columns,
                                 categorical_columns_with_nulls=train_categorical_null_columns,
                                 all_numeric_columns=numerical_columns,
                                 all_categorical_columns=categorical_columns,
                                 hyperparams=hyperparams,
                                 output_path=output_path,
                                 **imputation_kwargs))

    # Check if there are any nulls in the output
    assert not X_train_imputed.isnull().any().any(), "X_train_imputed contains null values"
    assert not X_test_imputed.isnull().any().any(), "X_test_imputed contains null values"


# Test if datawig returns same results with the same seed
def test_datawig_imputer_same_seed(acs_income_dataset_params, null_imputer_name,
                                   mcar_evaluation_scenario, common_seed, datawig_kwargs):
    # Init function variables
    dataset_name = ACS_INCOME_DATASET
    evaluation_scenario = mcar_evaluation_scenario
    experiment_seed = common_seed
    imputation_kwargs = datawig_kwargs

    train_injection_strategy, _ = parse_evaluation_scenario(evaluation_scenario)
    hyperparams = NULL_IMPUTERS_HYPERPARAMS.get(null_imputer_name, {}).get(dataset_name, {}).get(train_injection_strategy, {})

    (X_train_with_nulls, X_test_with_nulls,
     train_numerical_null_columns, train_categorical_null_columns,
     numerical_columns, categorical_columns) = acs_income_dataset_params

    # Apply datawig
    output_path = (pathlib.Path(__file__).parent.parent.parent.joinpath('results')
                       .joinpath("tests")
                       .joinpath(null_imputer_name)
                       .joinpath(dataset_name)
                       .joinpath(evaluation_scenario)
                       .joinpath(str(experiment_seed)))

    imputation_kwargs.update({'experiment_seed': experiment_seed})
    X_train_imputed1, X_test_imputed1, null_imputer_params_dct1 = (
        datawig_imputer.complete(X_train_with_nulls=X_train_with_nulls,
                                 X_test_with_nulls=X_test_with_nulls,
                                 numeric_columns_with_nulls=train_numerical_null_columns,
                                 categorical_columns_with_nulls=train_categorical_null_columns,
                                 all_numeric_columns=numerical_columns,
                                 all_categorical_columns=categorical_columns,
                                 hyperparams=hyperparams,
                                 output_path=output_path,
                                 **imputation_kwargs)
    )
    X_train_imputed2, X_test_imputed2, null_imputer_params_dct2 = (
        datawig_imputer.complete(X_train_with_nulls=X_train_with_nulls,
                                 X_test_with_nulls=X_test_with_nulls,
                                 numeric_columns_with_nulls=train_numerical_null_columns,
                                 categorical_columns_with_nulls=train_categorical_null_columns,
                                 all_numeric_columns=numerical_columns,
                                 all_categorical_columns=categorical_columns,
                                 hyperparams=hyperparams,
                                 output_path=output_path,
                                 **imputation_kwargs)
    )

    # Check if the results are identical
    assert X_train_imputed1.equals(X_train_imputed2), "X_train_imputed from datawig are not identical"
    assert X_test_imputed1.equals(X_test_imputed2), "X_test_imputed from datawig are not identical"
    assert_nested_dicts_equal(null_imputer_params_dct1, null_imputer_params_dct2,
                              assert_msg="null_imputer_params_dct from datawig are not identical")


# Test if datawig returns different results for different seeds
def test_datawig_imputer_diff_seed(acs_income_dataset_params, null_imputer_name,
                                   mcar_evaluation_scenario, common_seed, datawig_kwargs):
    # Init function variables
    dataset_name = ACS_INCOME_DATASET
    evaluation_scenario = mcar_evaluation_scenario
    experiment_seed = common_seed
    imputation_kwargs = datawig_kwargs

    train_injection_strategy, _ = parse_evaluation_scenario(evaluation_scenario)
    hyperparams = NULL_IMPUTERS_HYPERPARAMS.get(null_imputer_name, {}).get(dataset_name, {}).get(train_injection_strategy, {})

    (X_train_with_nulls, X_test_with_nulls,
     train_numerical_null_columns, train_categorical_null_columns,
     numerical_columns, categorical_columns) = acs_income_dataset_params

    # Apply datawig
    output_path = (pathlib.Path(__file__).parent.parent.parent.joinpath('results')
                   .joinpath("tests")
                   .joinpath(null_imputer_name)
                   .joinpath(dataset_name)
                   .joinpath(evaluation_scenario)
                   .joinpath(str(experiment_seed)))

    imputation_kwargs.update({'experiment_seed': 100})
    X_train_imputed1, X_test_imputed1, null_imputer_params_dct1 = (
        datawig_imputer.complete(X_train_with_nulls=X_train_with_nulls,
                                 X_test_with_nulls=X_test_with_nulls,
                                 numeric_columns_with_nulls=train_numerical_null_columns,
                                 categorical_columns_with_nulls=train_categorical_null_columns,
                                 all_numeric_columns=numerical_columns,
                                 all_categorical_columns=categorical_columns,
                                 hyperparams=hyperparams,
                                 output_path=output_path,
                                 **imputation_kwargs)
    )

    imputation_kwargs.update({'experiment_seed': 200})
    X_train_imputed2, X_test_imputed2, null_imputer_params_dct2 = (
        datawig_imputer.complete(X_train_with_nulls=X_train_with_nulls,
                                 X_test_with_nulls=X_test_with_nulls,
                                 numeric_columns_with_nulls=train_numerical_null_columns,
                                 categorical_columns_with_nulls=train_categorical_null_columns,
                                 all_numeric_columns=numerical_columns,
                                 all_categorical_columns=categorical_columns,
                                 hyperparams=hyperparams,
                                 output_path=output_path,
                                 **imputation_kwargs)
    )

    # Check if the results are identical
    assert not X_train_imputed1.equals(X_train_imputed2), "X_train_imputed from datawig is the same for different seeds"
    assert not X_test_imputed1.equals(X_test_imputed2), "X_test_imputed from datawig is the same for different seeds"
