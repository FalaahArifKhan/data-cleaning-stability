# Still More Shades of Null: An Evaluation Suite for Responsible Missing Value Imputation

This repository contains the source code, scripts, and datasets for the Shades-of-Null evaluation suit. The evaluation suit uses SOTA missing value imputation (MVI) techniques on a suite of novel evaluation settings on popular fairness benchmark datasets, including multi-mechanism missingness (when several different missingness patterns co-exist in the data) and missingness shift (when the missingness mechanism changes between development/training and deployment/testing), and using a large set of holistic evaluation metrics, including fairness and stability. The evaluation suit includes functionality for storing experiment results in a database, with MongoDB chosen for our purposes. Additionally, the evaluation suit is designed to be extensible, allowing researchers to incorporate custom datasets and apply new MVI techniques.


## Setup

Create a virtual environment and install requirements:
```shell
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
```dotenv
# Create configs/secrets.env file with database variables
DB_NAME=your_mongodb_name
CONNECTION_STRING=your_mongodb_connection_string
```

## Repository structure

* `source` directory contains code with custom classes for managing benchmark, database client, error injectors, null imputers, visualizations and some utils functions.
* `configs` directory contains all constants and configs for datasets, null imputers, ML models and evaluation scenarios.
* `scripts` directory contains main scripts for evaluating null imputers, baselines and ML models.
* `tests` directory contains tests covering the benchmark and null imputers.
* `notebooks` directory contains Jupyter notebooks with EDA and results visualization.
    * `cluster_analysis` subdirectory contains notebooks with analysis of the number of clusters in each dataset using silhoette scores and PCA, t-SNE, UMAP algorithms. Used to choose the correct number of clusters for the `clustering` null imputer.
    * `EDA` subdirectory contains notebooks with analysis of feature importance and feature correlation with the target for 6 datasets used in our experiments (_Section 3.1_ and _Appendix B.2_ in the paper).
    * `visualizations` subdirectory contains two subdirectories with visualisations for _imputation performance_ and _model performance_. Each of these subdirectories has the following structure:
      * `single_mechanism_exp` folder includes plots for single-mechanism missingness in both train and test sets (_Section 4.1, 4.2, 4.3_ and _Appendix C_ in the paper).
      * `multi_mechanism_exp` folder includes plots for multi-mechanism missingness in both train and test sets (_Section 4.1, 4.2, 4.3_ and _Appendix C_ in the paper).
      * `exp1` folder includes plots for missingness shift with a fixed error rate in both train and test sets (_Appendix D.1_ in the paper).
      * `exp2` folder includes plots for missingness shift with a variable error rate in the train set and a fixed error rate in the test set (_Section 5_ and _Appendix D_ in the paper).
      * `exp3` folder includes plots for missingness shift with a fixed error rate in the train set and a variable error rate in the test set (_Section 5_ and _Appendix D_ in the paper).
    * `Scatter_Plots.ipynb` notebook includes scatter plots for single-mechanism and multi-mechanism missingness colored by null imputers and shaped by datasets (_Section 4.4_ in the paper).
    * `Correlations.ipynb` notebook includes plots for spearman correlation between MVI technique, model type, test missingness, and performance metrics (F1, fairness and stability) for different train missingness mechanisms (_Section 6_ in the paper).


## Usage

### MVI technique evaluation

This console command evaluates single or multiple null imputation techniques on the selected dataset. The argument `evaluation_scenarios` defines which evaluation scenarios to use. Available scenarios are listed in `configs/scenarios_config.py`, but users have an option to create own evaluation scenarios. `tune_imputers` is a bool parameter whether to tune imputers or to reuse hyper-parameters from NULL_IMPUTERS_HYPERPARAMS in `configs/null_imputers_config.py`. `save_imputed_datasets` is a bool parameter whether to save imputed datasets locally for future use. `dataset` and `null_imputers` arguments should be chosen from supported datasets and  techniques. `run_nums` defines run numbers for different seeds, for example, the number 3 corresponds to 300 seed defined in EXPERIMENT_RUN_SEEDS in `configs/constants.py`.
```shell
python ./scripts/impute_nulls_with_predictor.py \
    --dataset folk \MVI
    --null_imputers [\"miss_forest\",\"datawig\"] \
    --run_nums [1,2,3] \
    --tune_imputers true \
    --save_imputed_datasets true \
    --evaluation_scenarios [\"exp1_mcar3\"]
```

### Models evaluation

This console command evaluates single or multiple null imputation techniques along with ML models training on the selected dataset. Arguments `evaluation_scenarios`, `dataset`, `null_imputers`, `run_nums` are used for the same purpose as in `impute_nulls_with_predictor.py`. `models` defines which ML models to evaluate in the pipeline. `ml_impute` is a bool argument which decides whether to impute null dynamically or use precomputed saved datasets with imputed values (if they are available).
```shell
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

This console command evaluates ML models on clean datasets (without injected nulls) for getting baseline metrics. Arguments follow same logic as in `evaluate_models.py`.
```shell
python ./scripts/evaluate_baseline.py \
    --dataset folk \
    --models [\"lr_clf\",\"mlp_clf\"] \
    --run_nums [1,2,3]
```


## Extending the benchmark

### Adding a new dataset

1. To add a new dataset, you need to use Virny wrapper BaseFlowDataset, where reading and basic preprocessing take place
   ([link to documentation](https://dataresponsibly.github.io/Virny/examples/Multiple_Models_Interface_Use_Case/#preprocess-the-dataset-and-create-a-baseflowdataset-class)).
2. Create a `config yaml` file in `configs/yaml_files` with settings for the number of estimators, bootstrap fraction and sensitive attributes dict like in example below.
```yaml
dataset_name: folk
bootstrap_fraction: 0.8
n_estimators: 50
computation_mode: error_analysis
sensitive_attributes_dct: {'SEX': '2', 'RAC1P': ['2', '3', '4', '5', '6', '7', '8', '9'], 'SEX & RAC1P': None}
```
3. In `configs/dataset_config.py`, add a newly created wrapper for your dataset specifing kwarg arguments, test set fraction and config yaml path in the `DATASET_CONFIG` dict.


### Adding a new ML model

1. To add a new model, add the model name to `MLModels` enum in `configs/constants.py`.
2. Set up a model instance and hyper-parameters grid for tuning inside the function `get_models_params_for_tuning` in `configs/models_config_for_tuning.py`. Model instance should inherit sklearn BaseEstimator from scikit-learn in order to support logic with tuning and fitting model ([link to documentation](https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html)).


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
3. Add your imputer name to the _ErrorRepairMethod_ enum in `configs/constants.py`.
4. [Optional] If a standard imputation pipeline does not work for a new null imputer, add a new if-statement to `source/custom_classes/benchmark.py` to the _impute_nulls method.


### Adding a new evaluation scenario

1. Add a configuration for the new _missingness scenario_ and the desired dataset to the `ERROR_INJECTION_SCENARIOS_CONFIG` dict in `configs/scenarios_config.py`. Missingness scenario should follow the structure below: `missing_features` are columns for null injection, and `setting` is a dict, specifying error rates and conditions for error injection.
```python
ACS_INCOME_DATASET: {
    "MCAR": [
        {
            'missing_features': ['WKHP', 'AGEP', 'SCHL', 'MAR'],
            'setting': {'error_rates': [0.1, 0.2, 0.3, 0.4, 0.5]},
        },
    ],
    "MAR": [
        {
            'missing_features': ['WKHP', 'SCHL'],
            'setting': {'condition': ('SEX', '2'), 'error_rates': [0.08, 0.12, 0.20, 0.28, 0.35]}
        }
    ],
    ...
}
```

2. Create a new _evaluation scenario_ with the new _missingness scenario_ in the `EVALUATION_SCENARIOS_CONFIG` dict in `configs/scenarios_config.py`. A new _missingness scenario_ can be used alone or combined with others. `train_injection_scenario` and `test_injection_scenarios` define settings of error injection for train and test sets, respectively. `test_injection_scenarios` takes a list as an input since the benchmark has an optimisation for multiple test sets.
```python
EVALUATION_SCENARIOS_CONFIG = {
    'mixed_exp': {
        'train_injection_scenario': 'MCAR1 & MAR1 & MNAR1',
        'test_injection_scenarios': ['MCAR1 & MAR1 & MNAR1'],
    },
    'exp1_mcar3': {
        'train_injection_scenario': 'MCAR3',
        'test_injection_scenarios': ['MCAR3', 'MAR3', 'MNAR3'],
    },
    ...
}
```
