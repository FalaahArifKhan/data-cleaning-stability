import copy
import pandas as pd
from sklearn.impute import SimpleImputer


def impute_with_median_mode(X_train_with_nulls: pd.DataFrame, X_test_with_nulls: pd.DataFrame,
                            train_numerical_null_columns: list, train_categorical_null_columns: list):
    X_train_imputed = copy.deepcopy(X_train_with_nulls)
    X_test_imputed = copy.deepcopy(X_test_with_nulls)

    # Impute with median
    median_imputer = SimpleImputer(strategy='median')
    X_train_imputed[train_numerical_null_columns] = median_imputer.fit_transform(X_train_imputed[train_numerical_null_columns])
    X_test_imputed[train_numerical_null_columns] = median_imputer.transform(X_test_imputed[train_numerical_null_columns])
    numerical_null_imputer_params = None

    # Impute with mode
    mode_imputer = SimpleImputer(strategy='most_frequent')
    X_train_imputed[train_categorical_null_columns] = mode_imputer.fit_transform(X_train_imputed[train_categorical_null_columns])
    X_test_imputed[train_categorical_null_columns] = mode_imputer.transform(X_test_imputed[train_categorical_null_columns])
    categorical_null_imputer_params = None

    return X_train_imputed, X_test_imputed, numerical_null_imputer_params, categorical_null_imputer_params
