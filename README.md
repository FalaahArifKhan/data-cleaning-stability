# data-cleaning-stability

Studying the impact of data cleaning techniques on fairness and stability


## Setup

Install datawig:
```shell
pip install mxnet-cu110
pip install datawig --no-deps
```


## Extending the benchmark

### Adding a new null imputer

1. Create a new imputation method for your imputer in `source/null_imputers/imputation_methods.py` similar to:
```python
def new_imputation_method(X_train_with_nulls: pd.DataFrame, X_tests_with_nulls_lst: list,
                          numeric_columns_with_nulls: list, categorical_columns_with_nulls: list,
                          hyperparams: dict, **kwargs):
    """
    This method imputes nulls using the new null imputer method.
    
    Arguments:
        X_train_with_nulls -- a training features df with nulls in numeric_columns_with_nulls and categorical_columns_with_nulls columns
        X_tests_with_nulls_lst -- a list of different X test dfs with nulls in numeric_columns_with_nulls and categorical_columns_with_nulls columns
        numeric_columns_with_nulls -- a list of numerical column names with nulls
        categorical_columns_with_nulls -- a list of categorical column names with nulls
        hyperparams -- a dictionary of tuned hyperparams for the null imputer
        kwargs -- all other params needed for the null imputer
    
    Returns:
        X_train_imputed (pd.DataFrame) -- a training features df with imputed columns defined in numeric_columns_with_nulls
                                          and categorical_columns_with_nulls
        X_tests_imputed_lst (pd.DataFrame) -- a training features df with imputed columns defined in numeric_columns_with_nulls 
                                         and categorical_columns_with_nulls
        null_imputer_params_dct (dict) -- a dictionary where a keys is a column name with nulls, and 
                                          a value is a dictionary of null imputer parameters used to impute this column
    """
    
    # Write here either a call to the algorithm or the algorithm itself
    ...
    
    return X_train_imputed, X_tests_imputed_lst, null_imputer_params_dct
```

2. Add the configuration of your new imputer to `configs/null_imputers_config.py` to the _NULL_IMPUTERS_CONFIG_ dictionary.
3. Add your name imputer name to the _ErrorRepairMethod_ enum in `configs/constants.py`.
4. [Optional] If a standard imputation pipeline does not work for a new null imputer, add a new if-statement to `source/custom_classes/benchmark.py` to the _impute_nulls method.
