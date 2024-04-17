import os
import copy
import uuid
import tqdm
import pathlib
import pandas as pd
from pprint import pprint
from datetime import datetime, timezone

from sklearn.metrics import mean_squared_error, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from virny.utils.custom_initializers import create_config_obj
from virny.user_interfaces.multiple_models_with_db_writer_api import compute_metrics_with_db_writer
from virny.user_interfaces.multiple_models_with_multiple_test_sets_api import compute_metrics_with_multiple_test_sets

from configs.models_config_for_tuning import get_models_params_for_tuning
from configs.null_imputers_config import NULL_IMPUTERS_CONFIG, NULL_IMPUTERS_HYPERPARAMS
from configs.constants import (EXP_COLLECTION_NAME, MODEL_HYPER_PARAMS_COLLECTION_NAME,
                               IMPUTATION_PERFORMANCE_METRICS_COLLECTION_NAME,
                               EXPERIMENT_RUN_SEEDS, NUM_FOLDS_FOR_TUNING, ErrorRepairMethod, ErrorInjectionStrategy)
from configs.datasets_config import DATASET_CONFIG
from configs.scenarios_config import ERROR_INJECTION_SCENARIOS_CONFIG
from source.utils.custom_logger import get_logger
from source.utils.model_tuning_utils import tune_ML_models
from source.utils.common_helpers import (generate_guid, create_base_flow_dataset,
                                         create_virny_base_flow_datasets, get_injection_scenarios)
from source.utils.dataframe_utils import preprocess_base_flow_dataset, preprocess_mult_base_flow_datasets
from source.custom_classes.database_client import DatabaseClient
from source.error_injectors.nulls_injector import NullsInjector
from source.validation import is_in_enum


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
        self.virny_config = create_config_obj(DATASET_CONFIG[dataset_name]['virny_config_path'])
        self.dataset_sensitive_attrs = [col for col in self.virny_config.sensitive_attributes_dct.keys() if '&' not in col]
        self.init_data_loader = DATASET_CONFIG[dataset_name]['data_loader'](**DATASET_CONFIG[dataset_name]['data_loader_kwargs'])

        self.__logger = get_logger()
        self.__db = DatabaseClient()
        # Create a unique uuid per session to manipulate in the database
        # by all experimental results generated in this session
        self._session_uuid = str(uuid.uuid1())
        print('Session UUID for all results of experiments in the current benchmark session:', self._session_uuid)

    def _split_dataset(self, data_loader, experiment_seed: int):
        if self.test_set_fraction < 0.0 or self.test_set_fraction > 1.0:
            raise ValueError("test_set_fraction must be a float in the [0.0-1.0] range")

        X_train_val, X_test, y_train_val, y_test = train_test_split(data_loader.X_data, data_loader.y_data,
                                                                    test_size=self.test_set_fraction,
                                                                    random_state=experiment_seed)
        return X_train_val, X_test, y_train_val, y_test

    def _remove_sensitive_attrs(self, X_train_val: pd.DataFrame, X_tests_lst: list, data_loader):
        X_train_val_wo_sensitive_attrs = X_train_val.drop(self.dataset_sensitive_attrs, axis=1)
        X_tests_wo_sensitive_attrs_lst = list(map(
            lambda X_test: X_test.drop(self.dataset_sensitive_attrs, axis=1),
            X_tests_lst
        ))
        numerical_columns_wo_sensitive_attrs = [col for col in data_loader.numerical_columns if col not in self.dataset_sensitive_attrs]
        categorical_columns_wo_sensitive_attrs = [col for col in data_loader.categorical_columns if col not in self.dataset_sensitive_attrs]

        return (X_train_val_wo_sensitive_attrs, X_tests_wo_sensitive_attrs_lst,
                numerical_columns_wo_sensitive_attrs, categorical_columns_wo_sensitive_attrs)

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

        # Save tunes parameters in database
        date_time_str = datetime.now(timezone.utc)
        tuned_params_df['Model_Best_Params'] = tuned_params_df['Model_Best_Params']
        tuned_params_df['Model_Tuning_Guid'] = tuned_params_df['Model_Name'].apply(
            lambda model_name: generate_guid(ordered_hierarchy_lst=[self.dataset_name, null_imputer_name,
                                                                    evaluation_scenario, experiment_seed, model_name])
        )
        self.__db.write_pandas_df_into_db(collection_name=MODEL_HYPER_PARAMS_COLLECTION_NAME,
                                          df=tuned_params_df,
                                          custom_tbl_fields_dct={
                                              'exp_pipeline_guid': generate_guid(ordered_hierarchy_lst=[self.dataset_name, null_imputer_name, evaluation_scenario, experiment_seed]),
                                              'session_uuid': self._session_uuid,
                                              'null_imputer_name': null_imputer_name,
                                              'evaluation_scenario': evaluation_scenario,
                                              'experiment_seed': experiment_seed,
                                              'record_create_date_time': date_time_str,
                                          })
        self.__logger.info("Models are tuned and their hyper-params are saved into a database")

        return models_config

    def _inject_nulls_into_one_set(self, df: pd.DataFrame, injection_scenario: str, experiment_seed: int):
        injection_strategy, error_rate_str = injection_scenario[:-1], injection_scenario[-1]
        error_rate_idx = int(error_rate_str) - 1
        for scenario_for_dataset in ERROR_INJECTION_SCENARIOS_CONFIG[self.dataset_name][injection_strategy]:
            error_rate = scenario_for_dataset['setting']['error_rates'][error_rate_idx]
            condition = None if injection_strategy == ErrorInjectionStrategy.mcar.value else scenario_for_dataset['setting']['condition']
            nulls_injector = NullsInjector(seed=experiment_seed,
                                           strategy=injection_strategy,
                                           columns_with_nulls=scenario_for_dataset['missing_features'],
                                           null_percentage=error_rate,
                                           condition=condition)
            df = nulls_injector.fit_transform(df)

        return df

    def _inject_nulls(self, X_train_val: pd.DataFrame, X_test: pd.DataFrame, evaluation_scenario: str, experiment_seed: int):
        train_injection_scenario, test_injection_scenarios_lst = get_injection_scenarios(evaluation_scenario)

        X_train_val_with_nulls = self._inject_nulls_into_one_set(df=X_train_val,
                                                                 injection_scenario=train_injection_scenario,
                                                                 experiment_seed=experiment_seed)
        X_tests_with_nulls_lst = list(map(
                lambda test_injection_strategy: self._inject_nulls_into_one_set(df=X_test,
                                                                                injection_scenario=test_injection_strategy,
                                                                                experiment_seed=experiment_seed),
                test_injection_scenarios_lst
        ))
        self.__logger.info('Nulls are successfully injected')

        print("X_tests_with_nulls_lst[2]['AGEP'].isna().sum() --", X_tests_with_nulls_lst[2]['AGEP'].isna().sum())
        print("X_tests_with_nulls_lst[2]['SCHL'].isna().sum() --", X_tests_with_nulls_lst[2]['SCHL'].isna().sum())
        print("X_tests_with_nulls_lst[2]['MAR'].isna().sum() --", X_tests_with_nulls_lst[2]['MAR'].isna().sum())
        print("X_tests_with_nulls_lst[2]['COW'].isna().sum() --", X_tests_with_nulls_lst[2]['COW'].isna().sum())

        return X_train_val_with_nulls, X_tests_with_nulls_lst

    def _impute_nulls(self, X_train_with_nulls, X_tests_with_nulls_lst, null_imputer_name, evaluation_scenario,
                      experiment_seed, numerical_columns, categorical_columns, tune_imputers):
        if not is_in_enum(null_imputer_name, ErrorRepairMethod) or null_imputer_name not in NULL_IMPUTERS_CONFIG.keys():
            raise ValueError(f'{null_imputer_name} null imputer is not implemented')

        if tune_imputers:
            hyperparams = None
        else:
            train_injection_strategy, _ = get_injection_scenarios(evaluation_scenario)
            hyperparams = NULL_IMPUTERS_HYPERPARAMS.get(null_imputer_name, {}).get(self.dataset_name, {}).get(train_injection_strategy, {})

        # Use a method, kwargs, and hyperparams from NULL_IMPUTERS_CONFIG
        imputation_method = NULL_IMPUTERS_CONFIG[null_imputer_name]["method"]
        imputation_kwargs = NULL_IMPUTERS_CONFIG[null_imputer_name]["kwargs"]
        imputation_kwargs.update({'experiment_seed': experiment_seed})

        # TODO: Save a result imputed dataset in imputed_data_dict for each imputation technique
        train_set_cols_with_nulls = X_train_with_nulls.columns[X_train_with_nulls.isna().any()].tolist()
        train_numerical_null_columns = list(set(train_set_cols_with_nulls).intersection(numerical_columns))
        train_categorical_null_columns = list(set(train_set_cols_with_nulls).intersection(categorical_columns))

        imputation_start_time = datetime.now()
        if null_imputer_name == ErrorRepairMethod.datawig.value:
            output_path = (pathlib.Path(__file__).parent.parent.parent.joinpath('results')
                               .joinpath(null_imputer_name)
                               .joinpath(self.dataset_name)
                               .joinpath(evaluation_scenario)
                               .joinpath(str(experiment_seed)))
            X_train_imputed, X_tests_imputed_lst, null_imputer_params_dct = (
                imputation_method(X_train_with_nulls=X_train_with_nulls,
                                  X_tests_with_nulls_lst=X_tests_with_nulls_lst,
                                  numeric_columns_with_nulls=train_numerical_null_columns,
                                  categorical_columns_with_nulls=train_categorical_null_columns,
                                  all_numeric_columns=numerical_columns,
                                  all_categorical_columns=categorical_columns,
                                  hyperparams=hyperparams,
                                  output_path=output_path,
                                  **imputation_kwargs))

        elif null_imputer_name == ErrorRepairMethod.automl.value:
            output_path = (pathlib.Path(__file__).parent.parent.parent.joinpath('results')
                           .joinpath(null_imputer_name)
                           .joinpath(self.dataset_name)
                           .joinpath(evaluation_scenario)
                           .joinpath(str(experiment_seed)))
            imputation_kwargs.update({'directory': output_path})
            X_train_imputed, X_tests_imputed_lst, null_imputer_params_dct = (
                imputation_method(X_train_with_nulls=X_train_with_nulls,
                                  X_tests_with_nulls_lst=X_tests_with_nulls_lst,
                                  numeric_columns_with_nulls=train_numerical_null_columns,
                                  categorical_columns_with_nulls=train_categorical_null_columns,
                                  hyperparams=hyperparams,
                                  **imputation_kwargs))

        else:
            X_train_imputed, X_tests_imputed_lst, null_imputer_params_dct = (
                imputation_method(X_train_with_nulls=X_train_with_nulls,
                                  X_tests_with_nulls_lst=X_tests_with_nulls_lst,
                                  numeric_columns_with_nulls=train_numerical_null_columns,
                                  categorical_columns_with_nulls=train_categorical_null_columns,
                                  hyperparams=hyperparams,
                                  **imputation_kwargs))

        imputation_end_time = datetime.now()
        imputation_runtime = (imputation_end_time - imputation_start_time).total_seconds() / 60.0
        self.__logger.info('Nulls are successfully imputed')

        return X_train_imputed, X_tests_imputed_lst, null_imputer_params_dct, imputation_runtime

    def _evaluate_imputation(self, real, imputed, corrupted, numerical_columns, null_imputer_name, null_imputer_params_dct):
        columns_with_nulls = corrupted.columns[corrupted.isna().any()].tolist()
        metrics_df = pd.DataFrame(columns=('Dataset_Name', 'Null_Imputer_Name', 'Null_Imputer_Params',
                                           'Column_Type', 'Column_With_Nulls', 'RMSE', 'Precision', 'Recall', 'F1_Score'))
        for column_idx, column_name in enumerate(columns_with_nulls):
            column_type = 'numerical' if column_name in numerical_columns else 'categorical'

            indexes = corrupted[column_name].isna()
            true = real.loc[indexes, column_name]
            pred = imputed.loc[indexes, column_name]

            rmse = None
            precision = None
            recall = None
            f1 = None
            if column_type == 'numerical':
                null_imputer_params = null_imputer_params_dct[column_name] if null_imputer_params_dct is not None else None
                rmse = mean_squared_error(true, pred, squared=False)
                print('RMSE for {}: {:.2f}'.format(column_name, rmse))
                print()
            else:
                null_imputer_params = null_imputer_params_dct[column_name] if null_imputer_params_dct is not None else None
                precision, recall, f1, _ = precision_recall_fscore_support(true, pred, average="micro")
                print('Precision for {}: {:.2f}'.format(column_name, precision))
                print('Recall for {}: {:.2f}'.format(column_name, recall))
                print('F1 score for {}: {:.2f}'.format(column_name, f1))
                print()

            # Save imputation performance metric of the imputer in a dataframe
            metrics_df.loc[column_idx] = [self.dataset_name, null_imputer_name, null_imputer_params,
                                          column_type, column_name, rmse, precision, recall, f1]

        return metrics_df

    def _save_imputation_metrics_to_db(self, train_imputation_metrics_df: pd.DataFrame, test_imputation_metrics_dfs_lst: list,
                                       imputation_runtime: float, null_imputer_name: str, evaluation_scenario: str, experiment_seed: int):
        train_imputation_metrics_df['Imputation_Guid'] = train_imputation_metrics_df['Column_With_Nulls'].apply(
            lambda column_with_nulls: generate_guid(ordered_hierarchy_lst=[self.dataset_name, null_imputer_name,
                                                                           evaluation_scenario, experiment_seed,
                                                                           'X_train_val', column_with_nulls])
        )
        self.__db.write_pandas_df_into_db(collection_name=IMPUTATION_PERFORMANCE_METRICS_COLLECTION_NAME,
                                          df=train_imputation_metrics_df,
                                          custom_tbl_fields_dct={
                                              'exp_pipeline_guid': generate_guid(ordered_hierarchy_lst=[self.dataset_name, null_imputer_name, evaluation_scenario, experiment_seed]),
                                              'session_uuid': self._session_uuid,
                                              'evaluation_scenario': evaluation_scenario,
                                              'experiment_seed': experiment_seed,
                                              'dataset_part': 'X_train_val',
                                              'runtime_in_mins': imputation_runtime,
                                              'record_create_date_time': datetime.now(timezone.utc),
                                          })

        # Save imputation results into a database for each test set from the evaluation scenario
        _, test_injection_scenarios_lst = get_injection_scenarios(evaluation_scenario)
        test_record_create_date_time = datetime.now(timezone.utc)
        for test_set_idx, test_imputation_metrics_df in enumerate(test_imputation_metrics_dfs_lst):
            test_injection_scenario = test_injection_scenarios_lst[test_set_idx]
            test_imputation_metrics_df['Imputation_Guid'] = test_imputation_metrics_df['Column_With_Nulls'].apply(
                lambda column_with_nulls: generate_guid(ordered_hierarchy_lst=[self.dataset_name, null_imputer_name,
                                                                               evaluation_scenario, experiment_seed,
                                                                               f'X_test_{test_injection_scenario}',
                                                                               column_with_nulls])
            )
            self.__db.write_pandas_df_into_db(collection_name=IMPUTATION_PERFORMANCE_METRICS_COLLECTION_NAME,
                                              df=test_imputation_metrics_df,
                                              custom_tbl_fields_dct={
                                                  'exp_pipeline_guid': generate_guid(ordered_hierarchy_lst=[self.dataset_name, null_imputer_name, evaluation_scenario, experiment_seed]),
                                                  'session_uuid': self._session_uuid,
                                                  'evaluation_scenario': evaluation_scenario,
                                                  'experiment_seed': experiment_seed,
                                                  'dataset_part': f'X_test_{test_injection_scenario}',
                                                  'runtime_in_mins': imputation_runtime,
                                                  'record_create_date_time': test_record_create_date_time,
                                              })

        self.__logger.info("Performance metrics and tuned parameters of the null imputer are saved into a database")

    def _save_imputed_datasets_to_fs(self, X_train_val: pd.DataFrame, X_tests_lst: pd.DataFrame,
                                     null_imputer_name: str, evaluation_scenario: str, experiment_seed: int):
        save_sets_dir_path = (pathlib.Path(__file__).parent.parent.parent
                              .joinpath('results')
                              .joinpath(self.dataset_name)
                              .joinpath(null_imputer_name))
        os.makedirs(save_sets_dir_path, exist_ok=True)

        train_set_filename = f'imputed_{self.dataset_name}_{experiment_seed}_{evaluation_scenario}_{null_imputer_name}_X_train_val.csv'
        X_train_val.to_csv(os.path.join(save_sets_dir_path, train_set_filename),
                           sep=",",
                           columns=X_train_val.columns,
                           index=False)

        # Save each imputed test set in a local filesystem
        _, test_injection_scenarios_lst = get_injection_scenarios(evaluation_scenario)
        for test_set_idx, X_test in enumerate(X_tests_lst):
            test_injection_scenario = test_injection_scenarios_lst[test_set_idx]
            test_set_filename = f'imputed_{self.dataset_name}_{experiment_seed}_{evaluation_scenario}_{null_imputer_name}_X_test_{test_injection_scenario}.csv'
            X_test.to_csv(os.path.join(save_sets_dir_path, test_set_filename),
                          sep=",",
                          columns=X_test.columns,
                          index=False)

        self.__logger.info("Imputed train and test sets are saved locally")

    def inject_and_impute_nulls(self, data_loader, null_imputer_name: str, evaluation_scenario: str,
                                experiment_seed: int, tune_imputers: bool = True, save_imputed_datasets: bool = False):
        # Split the dataset
        X_train_val, X_test, y_train_val, y_test = self._split_dataset(data_loader, experiment_seed)

        # Inject nulls not into sensitive attributes
        X_train_val_with_nulls, X_tests_with_nulls_lst = self._inject_nulls(X_train_val=X_train_val,
                                                                            X_test=X_test,
                                                                            evaluation_scenario=evaluation_scenario,
                                                                            experiment_seed=experiment_seed)

        # Remove sensitive attributes from train and test sets with nulls to avoid their usage during imputation
        (X_train_val_with_nulls_wo_sensitive_attrs,
         X_tests_with_nulls_wo_sensitive_attrs_lst,
         numerical_columns_wo_sensitive_attrs,
         categorical_columns_wo_sensitive_attrs) = self._remove_sensitive_attrs(X_train_val=X_train_val_with_nulls,
                                                                                X_tests_lst=X_tests_with_nulls_lst,
                                                                                data_loader=data_loader)

        # Impute nulls
        (X_train_val_imputed_wo_sensitive_attrs, X_tests_imputed_wo_sensitive_attrs_lst, null_imputer_params_dct,
         imputation_runtime) = self._impute_nulls(X_train_with_nulls=X_train_val_with_nulls_wo_sensitive_attrs,
                                                  X_tests_with_nulls_lst=X_tests_with_nulls_wo_sensitive_attrs_lst,
                                                  null_imputer_name=null_imputer_name,
                                                  evaluation_scenario=evaluation_scenario,
                                                  experiment_seed=experiment_seed,
                                                  categorical_columns=categorical_columns_wo_sensitive_attrs,
                                                  numerical_columns=numerical_columns_wo_sensitive_attrs,
                                                  tune_imputers=tune_imputers)
        print('X_tests_imputed_wo_sensitive_attrs_lst[0].columns -- ', X_tests_imputed_wo_sensitive_attrs_lst[0].columns)

        if null_imputer_name == ErrorRepairMethod.deletion.value:
            # Skip evaluation of an imputed train set for the deletion null imputer
            train_imputation_metrics_df = pd.DataFrame(columns=['Column_With_Nulls'])
            # Subset y_train_val to align with X_train_val_imputed_wo_sensitive_attrs
            y_train_val = y_train_val.loc[X_train_val_imputed_wo_sensitive_attrs.index]

        else:
            # Evaluate imputation for train and test sets
            print('\n')
            self.__logger.info('Evaluating imputation for X_train_val...')
            train_imputation_metrics_df = self._evaluate_imputation(real=X_train_val,
                                                                    corrupted=X_train_val_with_nulls_wo_sensitive_attrs,
                                                                    imputed=X_train_val_imputed_wo_sensitive_attrs,
                                                                    numerical_columns=numerical_columns_wo_sensitive_attrs,
                                                                    null_imputer_name=null_imputer_name,
                                                                    null_imputer_params_dct=null_imputer_params_dct)
        print('\n')
        self.__logger.info('Evaluating imputation for X_test sets...')
        test_imputation_metrics_dfs_lst = list(map(
            lambda X_tests_for_evaluation: \
                self._evaluate_imputation(real=X_test,
                                          corrupted=X_tests_for_evaluation[0],
                                          imputed=X_tests_for_evaluation[1],
                                          numerical_columns=numerical_columns_wo_sensitive_attrs,
                                          null_imputer_name=null_imputer_name,
                                          null_imputer_params_dct=null_imputer_params_dct),
            zip(X_tests_with_nulls_wo_sensitive_attrs_lst, X_tests_imputed_wo_sensitive_attrs_lst)
        ))

        # Save performance metrics and tuned parameters of the null imputer in database
        self._save_imputation_metrics_to_db(train_imputation_metrics_df=train_imputation_metrics_df,
                                            test_imputation_metrics_dfs_lst=test_imputation_metrics_dfs_lst,
                                            imputation_runtime=imputation_runtime,
                                            null_imputer_name=null_imputer_name,
                                            evaluation_scenario=evaluation_scenario,
                                            experiment_seed=experiment_seed)

        if save_imputed_datasets:
            self._save_imputed_datasets_to_fs(X_train_val=X_train_val_imputed_wo_sensitive_attrs,
                                              X_tests_lst=X_tests_imputed_wo_sensitive_attrs_lst,
                                              null_imputer_name=null_imputer_name,
                                              evaluation_scenario=evaluation_scenario,
                                              experiment_seed=experiment_seed)

        # Create a base flow dataset for Virny to compute metrics
        main_base_flow_dataset, extra_base_flow_datasets = \
            create_virny_base_flow_datasets(data_loader=data_loader,
                                            dataset_sensitive_attrs=self.dataset_sensitive_attrs,
                                            X_train_val_wo_sensitive_attrs=X_train_val_imputed_wo_sensitive_attrs,
                                            X_tests_wo_sensitive_attrs_lst=X_tests_imputed_wo_sensitive_attrs_lst,
                                            y_train_val=y_train_val,
                                            y_test=y_test,
                                            numerical_columns_wo_sensitive_attrs=numerical_columns_wo_sensitive_attrs,
                                            categorical_columns_wo_sensitive_attrs=categorical_columns_wo_sensitive_attrs)

        return main_base_flow_dataset, extra_base_flow_datasets

    def load_imputed_train_test_sets(self, data_loader, null_imputer_name: str, evaluation_scenario: str,
                                     experiment_seed: int):
        if self.test_set_fraction < 0.0 or self.test_set_fraction > 1.0:
            raise ValueError("test_set_fraction must be a float in the [0.0-1.0] range")

        # Split the dataset
        y_train_val, y_test = train_test_split(data_loader.y_data,
                                               test_size=self.test_set_fraction,
                                               random_state=experiment_seed)

        # Read imputed train and test sets from save_sets_dir_path
        save_sets_dir_path = (pathlib.Path(__file__).parent.parent.parent
                                  .joinpath('results')
                                  .joinpath(self.dataset_name)
                                  .joinpath(null_imputer_name))

        train_set_filename = f'imputed_{self.dataset_name}_{experiment_seed}_{evaluation_scenario}_{null_imputer_name}_X_train_val.csv'
        X_train_val_imputed_wo_sensitive_attrs = pd.read_csv(os.path.join(save_sets_dir_path, train_set_filename), header=0)

        test_set_filename = f'imputed_{self.dataset_name}_{experiment_seed}_{evaluation_scenario}_{null_imputer_name}_X_test.csv'
        X_test_imputed_wo_sensitive_attrs = pd.read_csv(os.path.join(save_sets_dir_path, test_set_filename), header=0)

        # Create a base flow dataset for Virny to compute metrics
        numerical_columns_wo_sensitive_attrs = [col for col in data_loader.numerical_columns if col not in self.dataset_sensitive_attrs]
        categorical_columns_wo_sensitive_attrs = [col for col in data_loader.categorical_columns if col not in self.dataset_sensitive_attrs]
        base_flow_dataset = create_virny_base_flow_dataset(data_loader=data_loader,
                                                           dataset_sensitive_attrs=self.dataset_sensitive_attrs,
                                                           X_train_val_wo_sensitive_attrs=X_train_val_imputed_wo_sensitive_attrs,
                                                           X_test_wo_sensitive_attrs=X_test_imputed_wo_sensitive_attrs,
                                                           y_train_val=y_train_val,
                                                           y_test=y_test,
                                                           numerical_columns_wo_sensitive_attrs=numerical_columns_wo_sensitive_attrs,
                                                           categorical_columns_wo_sensitive_attrs=categorical_columns_wo_sensitive_attrs)

        return base_flow_dataset

    def _run_exp_iter_for_joint_cleaning_and_training(self, data_loader, experiment_seed: int, evaluation_scenario: str,
                                                      null_imputer_name: str, model_names: list, custom_table_fields_dct: dict):
        if len(model_names) > 0 and null_imputer_name == ErrorRepairMethod.cp_clean.value:
            self.__logger.warning(f'model_names argument is ignored for {ErrorRepairMethod.cp_clean.value} '
                                  f'since only KNN is supported for this null imputation method')

        # Split the dataset
        X_train_val, X_test, y_train_val, y_test = self._split_dataset(data_loader, experiment_seed)

        # Inject nulls not into sensitive attributes
        X_train_val_with_nulls, X_test_with_nulls = self._inject_nulls(X_train_val=X_train_val,
                                                                       X_test=X_test,
                                                                       evaluation_scenario=evaluation_scenario,
                                                                       experiment_seed=experiment_seed)

        # Remove sensitive attributes from train and test sets to avoid their usage during imputation
        (X_train_val_wo_sensitive_attrs,
         X_test_wo_sensitive_attrs,
         numerical_columns_wo_sensitive_attrs,
         categorical_columns_wo_sensitive_attrs) = self._remove_sensitive_attrs(X_train_val=X_train_val_with_nulls,
                                                                                X_test=X_test_with_nulls,
                                                                                data_loader=data_loader)
        (X_train_val_with_nulls_wo_sensitive_attrs,
         X_test_with_nulls_wo_sensitive_attrs,  _, _) = self._remove_sensitive_attrs(X_train_val=X_train_val_with_nulls,
                                                                                     X_test=X_test_with_nulls,
                                                                                     data_loader=data_loader)

        # Define a directory path to save an intermediate state
        save_dir = (pathlib.Path(__file__).parent.parent.parent.joinpath('results')
                        .joinpath(null_imputer_name)
                        .joinpath(self.dataset_name)
                        .joinpath(evaluation_scenario)
                        .joinpath(str(experiment_seed)))

        # Use a method, kwargs, and hyperparams from NULL_IMPUTERS_CONFIG
        joint_cleaning_and_training_func = NULL_IMPUTERS_CONFIG[null_imputer_name]["method"]
        imputation_kwargs = NULL_IMPUTERS_CONFIG[null_imputer_name]["kwargs"]
        imputation_kwargs.update({'save_dir': save_dir})

        # Create a wrapper for the input joint cleaning-and-training method
        # to conduct in-depth performance profiling with Virny
        (models_config,
         X_train_with_nulls,
         y_train) = joint_cleaning_and_training_func(X_train_val=X_train_val_wo_sensitive_attrs,
                                                     y_train_val=y_train_val,
                                                     X_train_val_with_nulls=X_train_val_with_nulls_wo_sensitive_attrs,
                                                     numerical_columns=numerical_columns_wo_sensitive_attrs,
                                                     categorical_columns=categorical_columns_wo_sensitive_attrs,
                                                     experiment_seed=experiment_seed,
                                                     **imputation_kwargs)

        # Create a base flow dataset for Virny to compute metrics
        base_flow_dataset = create_virny_base_flow_dataset(data_loader=data_loader,
                                                           dataset_sensitive_attrs=self.dataset_sensitive_attrs,
                                                           X_train_val_wo_sensitive_attrs=X_train_with_nulls,
                                                           X_test_wo_sensitive_attrs=X_test_with_nulls,
                                                           y_train_val=y_train,
                                                           y_test=y_test,
                                                           numerical_columns_wo_sensitive_attrs=numerical_columns_wo_sensitive_attrs,
                                                           categorical_columns_wo_sensitive_attrs=categorical_columns_wo_sensitive_attrs)

        # Compute metrics for tuned models
        # TODO: use multiple test sets interface
        compute_metrics_with_db_writer(dataset=base_flow_dataset,
                                       config=self.virny_config,
                                       models_config=models_config,
                                       custom_tbl_fields_dct=custom_table_fields_dct,
                                       db_writer_func=self.__db.get_db_writer(collection_name=EXP_COLLECTION_NAME),
                                       with_predict_proba=False,  # joint cleaning-and-training models do not support a predict_proba method
                                       notebook_logs_stdout=False,
                                       verbose=0)

    def _run_exp_iter_for_standard_imputation(self, data_loader, experiment_seed: int, evaluation_scenario: str,
                                              null_imputer_name: str, model_names: list, tune_imputers: bool,
                                              ml_impute: bool, custom_table_fields_dct: dict):
        if ml_impute:
            main_base_flow_dataset, extra_base_flow_datasets = self.inject_and_impute_nulls(data_loader=data_loader,
                                                                                            null_imputer_name=null_imputer_name,
                                                                                            evaluation_scenario=evaluation_scenario,
                                                                                            tune_imputers=tune_imputers,
                                                                                            experiment_seed=experiment_seed)
        else:
            # TODO: extract train and test sets from AWS S3
            main_base_flow_dataset, extra_base_flow_datasets = self.load_imputed_train_test_sets(data_loader=data_loader,
                                                                                                 null_imputer_name=null_imputer_name,
                                                                                                 evaluation_scenario=evaluation_scenario,
                                                                                                 experiment_seed=experiment_seed)

        # Preprocess the dataset using the defined preprocessor
        main_base_flow_dataset, extra_test_sets = preprocess_mult_base_flow_datasets(main_base_flow_dataset, extra_base_flow_datasets)

        # Tune ML models
        models_config = self._tune_ML_models(model_names=model_names,
                                             base_flow_dataset=main_base_flow_dataset,
                                             experiment_seed=experiment_seed,
                                             evaluation_scenario=evaluation_scenario,
                                             null_imputer_name=null_imputer_name)

        # Compute metrics for tuned models
        # TODO: set model seed before passing to virny
        compute_metrics_with_multiple_test_sets(dataset=main_base_flow_dataset,
                                                extra_test_sets_lst=extra_test_sets,
                                                config=self.virny_config,
                                                models_config=models_config,
                                                custom_tbl_fields_dct=custom_table_fields_dct,
                                                db_writer_func=self.__db.get_db_writer(collection_name=EXP_COLLECTION_NAME),
                                                notebook_logs_stdout=False,
                                                verbose=0)

    def _run_exp_iter(self, init_data_loader, run_num, evaluation_scenario, null_imputer_name,
                      model_names, tune_imputers, ml_impute):
        data_loader = copy.deepcopy(init_data_loader)

        custom_table_fields_dct = dict()
        experiment_seed = EXPERIMENT_RUN_SEEDS[run_num - 1]
        custom_table_fields_dct['session_uuid'] = self._session_uuid
        custom_table_fields_dct['null_imputer_name'] = null_imputer_name
        custom_table_fields_dct['evaluation_scenario'] = evaluation_scenario
        custom_table_fields_dct['experiment_iteration'] = f'exp_iter_{run_num}'
        custom_table_fields_dct['dataset_split_seed'] = experiment_seed
        custom_table_fields_dct['model_init_seed'] = experiment_seed

        # Create exp_pipeline_guid to define a row level of granularity.
        # concat(exp_pipeline_guid, model_name, subgroup, metric) can be used to check duplicates of results
        # for the same experimental pipeline.
        custom_table_fields_dct['exp_pipeline_guid'] = (
            generate_guid(ordered_hierarchy_lst=[self.dataset_name, null_imputer_name, evaluation_scenario, experiment_seed]))

        self.__logger.info("Start an experiment iteration for the following custom params:")
        pprint(custom_table_fields_dct)
        print('\n', flush=True)

        if null_imputer_name in (ErrorRepairMethod.boost_clean.value, ErrorRepairMethod.cp_clean.value):
            self._run_exp_iter_for_joint_cleaning_and_training(data_loader=data_loader,
                                                               experiment_seed=experiment_seed,
                                                               evaluation_scenario=evaluation_scenario,
                                                               null_imputer_name=null_imputer_name,
                                                               model_names=model_names,
                                                               custom_table_fields_dct=custom_table_fields_dct)
        else:
            self._run_exp_iter_for_standard_imputation(data_loader=data_loader,
                                                       experiment_seed=experiment_seed,
                                                       evaluation_scenario=evaluation_scenario,
                                                       null_imputer_name=null_imputer_name,
                                                       model_names=model_names,
                                                       tune_imputers=tune_imputers,
                                                       ml_impute=ml_impute,
                                                       custom_table_fields_dct=custom_table_fields_dct)

    def run_experiment(self, run_nums: list, evaluation_scenarios: list, model_names: list, tune_imputers: bool, ml_impute: bool):
        self.__db.connect()

        total_iterations = len(self.null_imputers) * len(evaluation_scenarios) * len(run_nums)
        with tqdm.tqdm(total=total_iterations, desc="Experiment Progress") as pbar:
            for null_imputer_idx, null_imputer_name in enumerate(self.null_imputers):
                for evaluation_scenario_idx, evaluation_scenario in enumerate(evaluation_scenarios):
                    for run_idx, run_num in enumerate(run_nums):
                        self.__logger.info(f"{'=' * 30} NEW EXPERIMENT RUN {'=' * 30}")
                        print('Configs for a new experiment run:')
                        print(
                            f"Null imputer: {null_imputer_name} ({null_imputer_idx + 1} out of {len(self.null_imputers)})\n"
                            f"Evaluation scenario: {evaluation_scenario} ({evaluation_scenario_idx + 1} out of {len(evaluation_scenarios)})\n"
                            f"Run num: {run_num} ({run_idx + 1} out of {len(run_nums)})\n"
                        )
                        self._run_exp_iter(init_data_loader=self.init_data_loader,
                                           run_num=run_num,
                                           evaluation_scenario=evaluation_scenario,
                                           null_imputer_name=null_imputer_name,
                                           model_names=model_names,
                                           tune_imputers=tune_imputers,
                                           ml_impute=ml_impute)
                        pbar.update(1)
                        print('\n\n\n\n', flush=True)

        self.__db.close()
        self.__logger.info("Experimental results were successfully saved!")

    def _run_null_imputation_iter(self, init_data_loader, run_num, evaluation_scenario, null_imputer_name,
                                  tune_imputers, save_imputed_datasets):
        if null_imputer_name in (ErrorRepairMethod.boost_clean.value, ErrorRepairMethod.cp_clean.value):
            raise ValueError(f'To work with {ErrorRepairMethod.boost_clean.value} or {ErrorRepairMethod.cp_clean.value}, '
                             f'use scripts/evaluate_models.py')

        data_loader = copy.deepcopy(init_data_loader)
        experiment_seed = EXPERIMENT_RUN_SEEDS[run_num - 1]
        self.inject_and_impute_nulls(data_loader=data_loader,
                                     null_imputer_name=null_imputer_name,
                                     evaluation_scenario=evaluation_scenario,
                                     experiment_seed=experiment_seed,
                                     tune_imputers=tune_imputers,
                                     save_imputed_datasets=save_imputed_datasets)

    def impute_nulls_with_multiple_technique(self, run_nums: list, evaluation_scenarios: list,
                                             tune_imputers: bool, save_imputed_datasets: bool):
        self.__db.connect()

        total_iterations = len(self.null_imputers) * len(evaluation_scenarios) * len(run_nums)
        with tqdm.tqdm(total=total_iterations, desc="Null Imputation Progress") as pbar:
            for null_imputer_idx, null_imputer_name in enumerate(self.null_imputers):
                for evaluation_scenario_idx, evaluation_scenario in enumerate(evaluation_scenarios):
                    for run_idx, run_num in enumerate(run_nums):
                        self.__logger.info(f"{'=' * 30} NEW DATASET IMPUTATION RUN {'=' * 30}")
                        print('Configs for a new experiment run:')
                        print(
                            f"Null imputer: {null_imputer_name} ({null_imputer_idx + 1} out of {len(self.null_imputers)})\n"
                            f"Evaluation scenario: {evaluation_scenario} ({evaluation_scenario_idx + 1} out of {len(evaluation_scenarios)})\n"
                            f"Run num: {run_num} ({run_idx + 1} out of {len(run_nums)})\n"
                        )
                        self._run_null_imputation_iter(init_data_loader=self.init_data_loader,
                                                       run_num=run_num,
                                                       evaluation_scenario=evaluation_scenario,
                                                       null_imputer_name=null_imputer_name,
                                                       tune_imputers=tune_imputers,
                                                       save_imputed_datasets=save_imputed_datasets)
                        pbar.update(1)
                        print('\n\n\n\n', flush=True)

        self.__db.close()
        self.__logger.info("Experimental results were successfully saved!")

    def _prepare_baseline_dataset(self, data_loader, experiment_seed: int):
        # Split the dataset
        X_train_val, X_test, y_train_val, y_test = self._split_dataset(data_loader, experiment_seed)

        # Remove sensitive attributes from train and test sets with nulls to avoid their usage during model training
        (X_train_val_wo_sensitive_attrs,
         X_tests_wo_sensitive_attrs_lst,
         numerical_columns_wo_sensitive_attrs,
         categorical_columns_wo_sensitive_attrs) = self._remove_sensitive_attrs(X_train_val=X_train_val,
                                                                                X_tests_lst=[X_test],
                                                                                data_loader=data_loader)
        X_test_wo_sensitive_attrs = X_tests_wo_sensitive_attrs_lst[0]

        # Create a base flow dataset for Virny to compute metrics
        base_flow_dataset = create_base_flow_dataset(data_loader=data_loader,
                                                     dataset_sensitive_attrs=self.dataset_sensitive_attrs,
                                                     X_train_val_wo_sensitive_attrs=X_train_val_wo_sensitive_attrs,
                                                     X_test_wo_sensitive_attrs=X_test_wo_sensitive_attrs,
                                                     y_train_val=y_train_val,
                                                     y_test=y_test,
                                                     numerical_columns_wo_sensitive_attrs=numerical_columns_wo_sensitive_attrs,
                                                     categorical_columns_wo_sensitive_attrs=categorical_columns_wo_sensitive_attrs)

        return base_flow_dataset

    def _run_baseline_evaluation_iter(self, init_data_loader, run_num: int, model_names: list):
        null_imputer_name = 'baseline'
        evaluation_scenario = 'baseline'
        data_loader = copy.deepcopy(init_data_loader)

        custom_table_fields_dct = dict()
        experiment_seed = EXPERIMENT_RUN_SEEDS[run_num - 1]
        custom_table_fields_dct['session_uuid'] = self._session_uuid
        custom_table_fields_dct['null_imputer_name'] = null_imputer_name
        custom_table_fields_dct['evaluation_scenario'] = evaluation_scenario
        custom_table_fields_dct['experiment_iteration'] = f'exp_iter_{run_num}'
        custom_table_fields_dct['dataset_split_seed'] = experiment_seed
        custom_table_fields_dct['model_init_seed'] = experiment_seed

        # Create exp_pipeline_guid to define a row level of granularity.
        # concat(exp_pipeline_guid, model_name, subgroup, metric) can be used to check duplicates of results
        # for the same experimental pipeline.
        custom_table_fields_dct['exp_pipeline_guid'] = (
            generate_guid(ordered_hierarchy_lst=[self.dataset_name, null_imputer_name, evaluation_scenario, experiment_seed]))

        self.__logger.info("Start an experiment iteration for the following custom params:")
        pprint(custom_table_fields_dct)
        print('\n', flush=True)

        # Prepare and preprocess the dataset using the defined preprocessor
        base_flow_dataset = self._prepare_baseline_dataset(data_loader, experiment_seed)
        base_flow_dataset = preprocess_base_flow_dataset(base_flow_dataset)

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

    def evaluate_baselines(self, run_nums: list, model_names: list):
        self.__db.connect()

        total_iterations = len(run_nums)
        with tqdm.tqdm(total=total_iterations, desc="Baseline Evaluation Progress") as pbar:
            for run_idx, run_num in enumerate(run_nums):
                self.__logger.info(f"{'=' * 30} NEW BASELINE EVALUATION RUN {'=' * 30}")
                print('Configs for a new baseline evaluation run:')
                print(
                    f"Models: {model_names})\n"
                    f"Run num: {run_num} ({run_idx + 1} out of {len(run_nums)})\n"
                )
                self._run_baseline_evaluation_iter(init_data_loader=self.init_data_loader,
                                                   run_num=run_num,
                                                   model_names=model_names)
                pbar.update(1)
                print('\n\n\n\n', flush=True)

        self.__db.close()
        self.__logger.info("Performance metrics of the baselines were successfully saved!")
