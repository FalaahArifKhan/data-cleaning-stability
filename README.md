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

1. Create a new imputation method for your imputer in `source/null_imputers/imputation_methods.py`.
2. Add the configuration of your new imputer to `configs/null_imputers_config.py` to the _NULL_IMPUTERS_CONFIG_ dictionary.
3. Add your name imputer name to the _ErrorRepairMethod_ enum in `configs/constants.py`.
4. [Optional] If a standard imputation pipeline does not work for a new null imputer, add a new if-statement to `source/custom_classes/benchmark.py` to the _impute_nulls method.
