import copy
from pprint import pprint
from datetime import datetime, timezone

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

from virny.custom_classes.base_dataset import BaseFlowDataset
from virny.utils.custom_initializers import create_config_obj
from virny.user_interfaces.multiple_models_with_db_writer_api import compute_metrics_with_db_writer

from configs.models_config_for_tuning import get_models_params_for_tuning
from configs.constants import (EXP_COLLECTION_NAME, MODEL_HYPER_PARAMS_COLLECTION_NAME, EXPERIMENT_RUN_SEEDS,
                               NUM_FOLDS_FOR_TUNING, ErrorRepairMethod, ErrorInjectionStrategy)
from configs.datasets_config import DATASET_CONFIG
from configs.dataset_uuids import DATASET_UUIDS
from configs.evaluation_scenarios_config import EVALUATION_SCENARIOS_CONFIG
from source.utils.custom_logger import get_logger
from source.utils.model_tuning_utils import tune_ML_models
from source.custom_classes.database_client import DatabaseClient
from source.preprocessing import get_simple_preprocessor
from source.error_injectors.nulls_injector import NullsInjector


class Benchmark:
    """
    Class encapsulates all required methods to run different experiments
    """
    def __init__(self, dataset_name: str, null_imputers: list, model_names: list):
        """
        Constructor defining default variables
        """
        self.null_imputers = null_imputers
        self.model_names = model_names
        self.dataset_name = dataset_name

        self.num_folds_for_tuning = NUM_FOLDS_FOR_TUNING
        self.test_set_fraction = DATASET_CONFIG[dataset_name]['test_set_fraction']
        self.init_data_loader = DATASET_CONFIG[dataset_name]['data_loader'](**DATASET_CONFIG[dataset_name]['data_loader_kwargs'])
        self.virny_config = create_config_obj(DATASET_CONFIG[dataset_name]['virny_config_path'])
        self.dataset_sensitive_attrs = [col for col in self.virny_config.sensitive_attributes_dct.keys() if '&' not in col]

        self.__logger = get_logger()
        self.__db = DatabaseClient()

    def _tune_ML_models(self, model_names, base_flow_dataset, experiment_seed,
                        evaluation_scenario, null_imputer_name):
        # Get hyper-parameters for tuning. Each time reinitialize an init model and its hyper-params for tuning.
        all_models_params_for_tuning = get_models_params_for_tuning(experiment_seed)
        models_params_for_tuning = {model_name: all_models_params_for_tuning[model_name] for model_name in model_names}

        # Tune models and create a models config for metrics computation
        tuned_params_df, models_config = tune_ML_models(models_params_for_tuning=models_params_for_tuning,
                                                        base_flow_dataset=base_flow_dataset,
                                                        dataset_name=self.virny_config.dataset_name,
                                                        n_folds=self.num_folds_for_tuning)

        # Save tunes parameters
        date_time_str = datetime.now(timezone.utc)
        tuned_params_df['Model_Best_Params'] = tuned_params_df['Model_Best_Params'].astype(str)
        self.__db.write_pandas_df_into_db(collection_name=MODEL_HYPER_PARAMS_COLLECTION_NAME,
                                          df=tuned_params_df,
                                          custom_tbl_fields_dct={
                                              'session_uuid': DATASET_UUIDS[self.dataset_name][null_imputer_name],
                                              'experiment_seed': experiment_seed,
                                              'evaluation_scenario': evaluation_scenario,
                                              'null_imputer_name': null_imputer_name,
                                              'record_create_date_time': date_time_str,
                                          })
        self.__logger.info("Models are tuned and their hyper-params are saved into a database")

        return models_config

    def _inject_nulls_into_one_set(self, df: pd.DataFrame, injection_strategy: str, error_rate_idx: int, experiment_seed: int):
        for injection_scenario in EVALUATION_SCENARIOS_CONFIG[self.dataset_name][injection_strategy]:
            error_rate = injection_scenario['setting']['error_rates'][error_rate_idx]
            condition = None if injection_strategy == ErrorInjectionStrategy.mcar.value else injection_scenario['setting']['condition']
            nulls_injector = NullsInjector(seed=experiment_seed,
                                           strategy=injection_strategy,
                                           columns_with_nulls=injection_scenario['missing_features'],
                                           null_percentage=error_rate,
                                           condition=condition)
            df = nulls_injector.fit_transform(df)

        return df

    def _inject_nulls(self, X_train_val: pd.DataFrame, X_test: pd.DataFrame, evaluation_scenario: str, experiment_seed: int):
        evaluation_scenario = evaluation_scenario.upper()
        error_rate_idx = int(evaluation_scenario[-1]) - 1
        train_injection_strategy, test_injection_strategy = evaluation_scenario[:-1].split('_')

        X_train_val_with_nulls = self._inject_nulls_into_one_set(df=X_train_val,
                                                                 injection_strategy=train_injection_strategy,
                                                                 error_rate_idx=error_rate_idx,
                                                                 experiment_seed=experiment_seed)
        X_test_with_nulls = self._inject_nulls_into_one_set(df=X_test,
                                                            injection_strategy=test_injection_strategy,
                                                            error_rate_idx=error_rate_idx,
                                                            experiment_seed=experiment_seed)

        return X_train_val_with_nulls, X_test_with_nulls

    def _impute_nulls(self, X_train_with_nulls, X_test_with_nulls, null_imputer_name, experiment_seed,
                      categorical_columns, numerical_columns):
        # TODO:
        #  Save a result imputed dataset in imputed_data_dict for each imputation technique
        #  For each imputation technique:
        #  - Impute with the imputation technique
        #  - Measure imputation metrics
        #  - Make plots for other techniques except "drop-column", since we dropped the column based on this technique
        #  Save metrics of imputations techniques to a .json file for future analysis

        train_set_cols_with_nulls = X_train_with_nulls.columns[X_train_with_nulls.isna().any()].tolist()
        train_categorical_null_columns = list(set(train_set_cols_with_nulls).intersection(categorical_columns))
        train_numerical_null_columns = list(set(train_set_cols_with_nulls).intersection(numerical_columns))

        test_set_cols_with_nulls = X_test_with_nulls.columns[X_test_with_nulls.isna().any()].tolist()
        test_categorical_null_columns = list(set(test_set_cols_with_nulls).intersection(categorical_columns))
        test_numerical_null_columns = list(set(test_set_cols_with_nulls).intersection(numerical_columns))

        common_categorical_null_columns = list(set(train_categorical_null_columns).intersection(set(test_categorical_null_columns)))
        common_numerical_null_columns = list(set(train_numerical_null_columns).intersection(set(test_numerical_null_columns)))

        X_train_imputed = copy.deepcopy(X_train_with_nulls)
        X_test_imputed = copy.deepcopy(X_test_with_nulls)
        if null_imputer_name == ErrorRepairMethod.median_mode.value:
            # Impute with median
            median_imputer = SimpleImputer(strategy='median')
            X_train_imputed[common_numerical_null_columns] = median_imputer.fit_transform(X_train_imputed[common_numerical_null_columns])
            X_test_imputed[common_numerical_null_columns] = median_imputer.transform(X_test_imputed[common_numerical_null_columns])

            # Impute with mode
            mode_imputer = SimpleImputer(strategy='most_frequent')
            X_train_imputed[common_categorical_null_columns] = mode_imputer.fit_transform(X_train_imputed[common_categorical_null_columns])
            X_test_imputed[common_categorical_null_columns] = mode_imputer.transform(X_test_imputed[common_categorical_null_columns])

        else:
            raise ValueError(f'{null_imputer_name} null imputer is not implemented')

        return X_train_imputed, X_test_imputed

    def inject_and_impute_nulls(self, data_loader, null_imputer_name: str, evaluation_scenario: str, experiment_seed: int):
        if self.test_set_fraction < 0.0 or self.test_set_fraction > 1.0:
            raise ValueError("test_set_fraction must be a float in the [0.0-1.0] range")

        # Split and preprocess the dataset
        X_train_val, X_test, y_train_val, y_test = train_test_split(data_loader.X_data, data_loader.y_data,
                                                                    test_size=self.test_set_fraction,
                                                                    random_state=experiment_seed)
        # Inject nulls
        X_train_val_with_nulls, X_test_with_nulls = self._inject_nulls(X_train_val=X_train_val,
                                                                       X_test=X_test,
                                                                       evaluation_scenario=evaluation_scenario,
                                                                       experiment_seed=experiment_seed)
        # Impute nulls
        X_train_val_imputed, X_test_imputed = self._impute_nulls(X_train_with_nulls=X_train_val_with_nulls,
                                                                 X_test_with_nulls=X_test_with_nulls,
                                                                 null_imputer_name=null_imputer_name,
                                                                 experiment_seed=experiment_seed,
                                                                 categorical_columns=data_loader.categorical_columns,
                                                                 numerical_columns=data_loader.numerical_columns)

        return BaseFlowDataset(init_features_df=data_loader.full_df[self.dataset_sensitive_attrs],  # keep only sensitive attributes with original indexes to compute group metrics
                               X_train_val=X_train_val_imputed,
                               X_test=X_test_imputed,
                               y_train_val=y_train_val,
                               y_test=y_test,
                               target=data_loader.target,
                               numerical_columns=data_loader.numerical_columns,
                               categorical_columns=data_loader.categorical_columns)

    def _run_exp_iter(self, init_data_loader, run_num, evaluation_scenario, null_imputer_name, model_names, ml_impute):
        data_loader = copy.deepcopy(init_data_loader)

        custom_table_fields_dct = dict()
        experiment_seed = EXPERIMENT_RUN_SEEDS[run_num - 1]
        custom_table_fields_dct['session_uuid'] = DATASET_UUIDS[self.dataset_name][null_imputer_name]
        custom_table_fields_dct['null_imputer_name'] = null_imputer_name
        custom_table_fields_dct['evaluation_scenario'] = evaluation_scenario
        custom_table_fields_dct['experiment_iteration'] = f'exp_iter_{run_num}'
        custom_table_fields_dct['dataset_split_seed'] = experiment_seed
        custom_table_fields_dct['model_init_seed'] = experiment_seed

        self.__logger.info("Start an experiment iteration for the following custom params:")
        pprint(custom_table_fields_dct)
        print('\n', flush=True)

        if ml_impute:
            base_flow_dataset = self.inject_and_impute_nulls(data_loader=data_loader,
                                                             null_imputer_name=null_imputer_name,
                                                             evaluation_scenario=evaluation_scenario,
                                                             experiment_seed=experiment_seed)
        else:
            # TODO: extract train and test sets from AWS S3
            base_flow_dataset = None

        # Remove sensitive attributes to create a blind estimator
        base_flow_dataset.categorical_columns = [col for col in base_flow_dataset.categorical_columns if col not in self.dataset_sensitive_attrs]
        base_flow_dataset.numerical_columns = [col for col in base_flow_dataset.numerical_columns if col not in self.dataset_sensitive_attrs]
        base_flow_dataset.X_train_val = base_flow_dataset.X_train_val.drop(self.dataset_sensitive_attrs, axis=1)
        base_flow_dataset.X_test = base_flow_dataset.X_test.drop(self.dataset_sensitive_attrs, axis=1)

        # Preprocess the dataset using the defined preprocessor
        column_transformer = get_simple_preprocessor(base_flow_dataset)
        column_transformer = column_transformer.set_output(transform="pandas")  # Set transformer output to a pandas df
        base_flow_dataset.X_train_val = column_transformer.fit_transform(base_flow_dataset.X_train_val)
        base_flow_dataset.X_test = column_transformer.transform(base_flow_dataset.X_test)

        # Tune ML models
        models_config = self._tune_ML_models(model_names=model_names,
                                             base_flow_dataset=base_flow_dataset,
                                             experiment_seed=experiment_seed,
                                             evaluation_scenario=evaluation_scenario,
                                             null_imputer_name=null_imputer_name)

        # Compute metrics for tuned models
        compute_metrics_with_db_writer(dataset=base_flow_dataset,
                                       config=self.virny_config,
                                       models_config=models_config,
                                       custom_tbl_fields_dct=custom_table_fields_dct,
                                       db_writer_func=self.__db.get_db_writer(collection_name=EXP_COLLECTION_NAME),
                                       notebook_logs_stdout=False,
                                       verbose=0)

    def run_experiment(self, run_nums: list, evaluation_scenarios: list, model_names: list, ml_impute: bool):
        self.__db.connect()
        # TODO: add tqdm
        for null_imputer_name in self.null_imputers:
            for evaluation_scenario in evaluation_scenarios:
                for run_num in run_nums:
                    # TODO: apply strategy based on evaluation scenario
                    self._run_exp_iter(init_data_loader=self.init_data_loader,
                                       run_num=run_num,
                                       evaluation_scenario=evaluation_scenario,
                                       null_imputer_name=null_imputer_name,
                                       model_names=model_names,
                                       ml_impute=ml_impute)

        self.__db.close()
        self.__logger.info("Experimental results were successfully saved!")
