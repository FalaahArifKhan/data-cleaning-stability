import pytest
import pathlib
import numpy as np

import source.null_imputers.datawig_imputer as datawig_imputer
from source.validation import parse_evaluation_scenario
from configs.datasets_config import ACS_INCOME_DATASET
from configs.null_imputers_config import NULL_IMPUTERS_HYPERPARAMS

from tests import assert_nested_dicts_equal


@pytest.fixture(scope="function")
def null_imputer_name():
    return 'datawig'


@pytest.fixture(scope="function")
def datawig_kwargs():
    return {"precision_threshold": 0.0, "num_epochs": 1, "iterations": 1}


# Test if output of datawig does not contain nulls
def test_datawig_imputer_no_nulls(acs_income_dataset_params, null_imputer_name,
                                  mcar_mar_evaluation_scenario, common_seed, datawig_kwargs):
    import tensorflow as tf

    # Check if GPU support is available
    gpu_available = tf.config.list_physical_devices('GPU')

    if gpu_available:
        # Get the number of available GPUs
        num_gpus = len(gpu_available)
        print(f"GPU is available with {num_gpus} device(s)!")
    else:
        print("GPU is not available.")

    # Init function variables
    dataset_name = ACS_INCOME_DATASET
    evaluation_scenario = mcar_mar_evaluation_scenario
    experiment_seed = common_seed
    imputation_kwargs = datawig_kwargs

    train_injection_strategy, _ = parse_evaluation_scenario(evaluation_scenario)
    hyperparams = NULL_IMPUTERS_HYPERPARAMS.get(null_imputer_name, {}).get(dataset_name, {}).get(train_injection_strategy, {})

    (X_train_with_nulls, X_test_with_nulls,
     train_numerical_null_columns, train_categorical_null_columns,
     numerical_columns, categorical_columns) = acs_income_dataset_params

    # Apply datawig
    output_path = (pathlib.Path(__file__).parent.parent.parent.joinpath('results')
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
    assert not np.isnan(X_train_imputed).any(), "X_train_imputed contains null values"
    assert not np.isnan(X_test_imputed).any(), "X_test_imputed contains null values"


# Test if datawig returns same results with the same seed
def test_datawig_imputer_same_seed(acs_income_dataset_params, null_imputer_name,
                                   mcar_mar_evaluation_scenario, common_seed, datawig_kwargs):
    # Init function variables
    dataset_name = ACS_INCOME_DATASET
    evaluation_scenario = mcar_mar_evaluation_scenario
    experiment_seed = common_seed
    imputation_kwargs = datawig_kwargs

    train_injection_strategy, _ = parse_evaluation_scenario(evaluation_scenario)
    hyperparams = NULL_IMPUTERS_HYPERPARAMS.get(null_imputer_name, {}).get(dataset_name, {}).get(train_injection_strategy, {})

    (X_train_with_nulls, X_test_with_nulls,
     train_numerical_null_columns, train_categorical_null_columns,
     numerical_columns, categorical_columns) = acs_income_dataset_params

    # Apply datawig
    output_path = (pathlib.Path(__file__).parent.parent.parent.joinpath('results')
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
    np.testing.assert_allclose(
        X_train_imputed1, X_train_imputed2,
        atol=1e-9, rtol=1e-9, err_msg="X_train_imputed from datawig are not identical"
    )
    np.testing.assert_allclose(
        X_test_imputed1, X_test_imputed2,
        atol=1e-9, rtol=1e-9, err_msg="X_test_imputed from datawig are not identical"
    )
    assert_nested_dicts_equal(null_imputer_params_dct1, null_imputer_params_dct2,
                              assert_msg="null_imputer_params_dct from datawig are not identical")
