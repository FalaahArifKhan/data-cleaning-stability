# Fairness and Stability under Realistic Missingness and Missingness Shift: Results of a Large-Scale Empirical Study

This repository contains the source code, scripts, and datasets for "Fairness and Stability under Realistic Missingness and Missingness Shift" benchmark. Benchmark uses state-of-the-art MVM techniques on a suite of novel evaluation settings on popular fairness benchmark datasets, including multi-mechanism missingness (when several different missingness patterns co-exist in the data) and missingness shift (when the missingness mechanism changes between development/training and deployment/testing), and using a large set of holistic evaluation metrics, including fairness and stability. The benchmark includes functionality for storing experiment results in a database, with MongoDB chosen for our purposes. Additionally, the benchmark is designed to be extensible, allowing researchers to incorporate custom datasets and apply new MVM techniques.


## Setup

Create a virtual environment and install requirements:
```
python -m venv venv 
source venv/bin/activate
pip3 install --upgrade pip3
pip3 install -r requiremnents.txt
```

Install datawig:
```shell
pip3 install mxnet-cu110
pip3 install datawig --no-deps

# In case of an import error for libcuda.so, use the command below recommended in
# https://stackoverflow.com/questions/54249577/importerror-libcuda-so-1-cannot-open-shared-object-file
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.0/compat
```

Add MongoDB secrets (optional)
```
# Create configs/secrets.env file with database variables
DB_NAME=your_mongodb_name
CONNECTION_STRING=your_mongodb_connection_string
```

## Repository structure

* `source` directory contains code with custom classes for managing benchmark, database client, error injectors, null imputers, visualizations and some utils functions.
* `notebooks` directory contains Jupyter notebooks with EDA and results visualization.
* `configs` directory contains all constants and configs for datasets, null imputers, classifiers and evaluation scenarios.
* `tests` directory contains tests covering benchmark and null imputers.
* `scripts` directory contains main scripts for evaluating null imputers, baselines and models.


## Usage

### MVM technique evaluation

This console command evaluates single or multiple null imputation techniques on chosen dataset. The argument `evaluation_scenarios` defines which missingness scenario to use. Available scenarios are listed in `configs/scenarios_config.py`. `tune_imputers` is a bool parameter whether to tune imputers. `save_imputed_datasets` is a bool parameter whether to save locally imputed datasets for future use. `dataset` and `null_imputers` arguments should be chosen from supported datasets and MVM techniques. `run_nums` defines number of runs with different seeds.
```
python ./scripts/impute_nulls_with_predictor.py \
    --dataset folk \
    --null_imputers [\"miss_forest\",\"datawig\"] \
    --run_nums [1,2,3] \
    --tune_imputers true \
    --save_imputed_datasets true \
    --evaluation_scenarios [\"exp1_mcar3\"]
```

### Models evaluation

This console command evaluates single or multiple null imputation techniques along with classifiers training on chosen dataset. Arguments `evaluation_scenarios`, `dataset`, `null_imputers`, `run_nums` are used for same purpose as in `impute_nulls_with_predictor.py`. `models` defines which classifiers train in pipeline. `ml_impute` is a bool argument which decides whether to impute null dynamically or use precomputed saved datasets (if they are available).
```
python ./scripts/evaluate_models.py \
    --dataset folk \
    --null_imputers [\"miss_forest\",\"datawig\"] \
    --models [\"lr_clf\",\"mlp_clf\"] \
    --run_nums [1,2,3] \
    --tune_imputers true \
    --save_imputed_datasets true \
    --ml_impute true \
    --evaluation_scenarios [\"exp1_mcar3\"]
```

### Baseline evaluation

This console command evaluates classifiers on clean datasets (without injected nulls) for getting baseline metrics. Arguments follow same logic as in `evaluate_models.py`.
```
python ./scripts/evaluate_baseline.py \
    --dataset folk \
    --models [\"lr_clf\",\"mlp_clf\"] \
    --run_nums [1,2,3]
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
        X_tests_imputed_lst (list) -- a list of test features df with imputed columns defined in numeric_columns_with_nulls 
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
