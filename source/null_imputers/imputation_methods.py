import copy
import pandas as pd
from sklearn.impute import SimpleImputer


def impute_with_simple_imputer(X_train_with_nulls: pd.DataFrame, X_test_with_nulls: pd.DataFrame,
                               numeric_columns_with_nulls: list, categorical_columns_with_nulls: list,
                               hyperparams: dict, **kwargs):
    X_train_imputed = copy.deepcopy(X_train_with_nulls)
    X_test_imputed = copy.deepcopy(X_test_with_nulls)

    # Impute numerical columns
    median_imputer = SimpleImputer(strategy=kwargs['num'])
    X_train_imputed[numeric_columns_with_nulls] = median_imputer.fit_transform(X_train_imputed[numeric_columns_with_nulls])
    X_test_imputed[numeric_columns_with_nulls] = median_imputer.transform(X_test_imputed[numeric_columns_with_nulls])

    # Impute categorical columns
    mode_imputer = SimpleImputer(strategy=kwargs['cat'])
    X_train_imputed[categorical_columns_with_nulls] = mode_imputer.fit_transform(X_train_imputed[categorical_columns_with_nulls])
    X_test_imputed[categorical_columns_with_nulls] = mode_imputer.transform(X_test_imputed[categorical_columns_with_nulls])

    null_imputer_params_dct = None
    return X_train_imputed, X_test_imputed, null_imputer_params_dct
