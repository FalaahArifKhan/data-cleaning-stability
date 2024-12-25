import os
import uuid
import shutil
import pathlib
import pytorch_tabular
import pandas as pd

from datetime import datetime, timezone
from sklearn.metrics import mean_squared_error, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from virny.utils.custom_initializers import create_config_obj
from virny.utils import create_test_protected_groups

from configs.models_config_for_tuning import get_models_params_for_tuning
from configs.null_imputers_config import NULL_IMPUTERS_CONFIG, NULL_IMPUTERS_HYPERPARAMS
from configs.constants import (MODEL_HYPER_PARAMS_COLLECTION_NAME, IMPUTATION_PERFORMANCE_METRICS_COLLECTION_NAME,
                               NUM_FOLDS_FOR_TUNING, ErrorRepairMethod, ErrorInjectionStrategy)
from configs.datasets_config import DATASET_CONFIG
from configs.scenarios_config import ERROR_INJECTION_SCENARIOS_CONFIG
from source.utils.custom_logger import get_logger
from source.utils.dataframe_utils import calculate_kl_divergence
from source.utils.model_tuning_utils import tune_sklearn_models, tune_pytorch_tabular_models
from source.utils.common_helpers import (generate_guid, create_base_flow_dataset, get_injection_scenarios)
from source.custom_classes.database_client import DatabaseClient, get_secrets_path
from source.error_injectors.nulls_injector import NullsInjector
from source.validation import is_in_enum


class MLLifecycle:
    """
    Class encapsulates all required ML lifecycle steps to run different experiments
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

        self._logger = get_logger()
        self._db = DatabaseClient(secrets_path=get_secrets_path('secrets.env'))
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

        # Separate models on sklearn API and pytorch tabular API
        sklearn_models_for_tuning = dict()
        pytorch_tabular_models_for_tuning = dict()
        for model_name in model_names:
            model_obj = all_models_params_for_tuning[model_name]['model']
            if isinstance(model_obj, pytorch_tabular.config.ModelConfig):
                pytorch_tabular_models_for_tuning[model_name] = all_models_params_for_tuning[model_name]
            else:
                sklearn_models_for_tuning[model_name] = all_models_params_for_tuning[model_name]

        # Tune models and create a models config for metrics computation
        sklearn_tuned_params_df, sklearn_models_config = pd.DataFrame(), dict()
        if len(sklearn_models_for_tuning) > 0:
            sklearn_tuned_params_df, sklearn_models_config = tune_sklearn_models(models_params_for_tuning=sklearn_models_for_tuning,
                                                                                 base_flow_dataset=base_flow_dataset,
                                                                                 dataset_name=self.virny_config.dataset_name,
                                                                                 n_folds=self.num_folds_for_tuning)

        pytorch_tabular_tuned_params_df, pytorch_tabular_models_config = pd.DataFrame(), dict()
        if len(pytorch_tabular_models_for_tuning) > 0:
            pytorch_tabular_tuned_params_df, pytorch_tabular_models_config = \
                tune_pytorch_tabular_models(models_params_for_tuning=pytorch_tabular_models_for_tuning,
                                            base_flow_dataset=base_flow_dataset,
                                            dataset_name=self.virny_config.dataset_name,
                                            null_imputer_name=null_imputer_name,
                                            evaluation_scenario=evaluation_scenario,
                                            experiment_seed=experiment_seed)

        # Save tunes parameters in database
        models_config = {**sklearn_models_config, **pytorch_tabular_models_config}
        tuned_params_df = pd.concat([sklearn_tuned_params_df, pytorch_tabular_tuned_params_df])
        date_time_str = datetime.now(timezone.utc)
        tuned_params_df['Model_Tuning_Guid'] = tuned_params_df['Model_Name'].apply(
            lambda model_name: generate_guid(ordered_hierarchy_lst=[self.dataset_name, null_imputer_name,
                                                                    evaluation_scenario, experiment_seed, model_name])
        )
        self._db.write_pandas_df_into_db(collection_name=MODEL_HYPER_PARAMS_COLLECTION_NAME,
                                         df=tuned_params_df,
                                         custom_tbl_fields_dct={
                                             'exp_pipeline_guid': generate_guid(ordered_hierarchy_lst=[self.dataset_name, null_imputer_name, evaluation_scenario, experiment_seed]),
                                             'session_uuid': self._session_uuid,
                                             'null_imputer_name': null_imputer_name,
                                             'evaluation_scenario': evaluation_scenario,
                                             'experiment_seed': experiment_seed,
                                             'record_create_date_time': date_time_str,
                                         })
        self._logger.info("Models are tuned and their hyper-params are saved into a database")

        return models_config

    def _inject_nulls_into_one_set_with_single_scenario(self, df: pd.DataFrame, injection_scenario: str, experiment_seed: int):
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

    def _inject_nulls_into_one_set(self, df: pd.DataFrame, injection_scenario: str, experiment_seed: int):
        if '&' in injection_scenario:
            single_injection_scenarios = [s.strip() for s in injection_scenario.split('&')]
            for single_injection_scenario in single_injection_scenarios:
                df = self._inject_nulls_into_one_set_with_single_scenario(df=df,
                                                                          injection_scenario=single_injection_scenario,
                                                                          experiment_seed=experiment_seed)
        else:
            df = self._inject_nulls_into_one_set_with_single_scenario(df=df,
                                                                      injection_scenario=injection_scenario,
                                                                      experiment_seed=experiment_seed)

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
        self._logger.info('Nulls are successfully injected')

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
        imputation_kwargs.update({'dataset_name': self.dataset_name})

        train_set_cols_with_nulls = X_train_with_nulls.columns[X_train_with_nulls.isna().any()].tolist()
        train_numerical_null_columns = list(set(train_set_cols_with_nulls).intersection(numerical_columns))
        train_categorical_null_columns = list(set(train_set_cols_with_nulls).intersection(categorical_columns))

        imputation_start_time = datetime.now()
        if null_imputer_name == ErrorRepairMethod.datawig.value:
            output_path = (pathlib.Path(__file__).parent.parent.parent
                           .joinpath('results')
                           .joinpath('intermediate_state')
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
            # Remove all files created by datawig to save storage space
            shutil.rmtree(output_path)

        elif null_imputer_name == ErrorRepairMethod.automl.value:
            output_path = (pathlib.Path(__file__).parent.parent.parent
                           .joinpath('results')
                           .joinpath('intermediate_state')
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
            # Remove all files created by automl to save storage space
            shutil.rmtree(output_path)

        elif null_imputer_name in (ErrorRepairMethod.gain.value, ErrorRepairMethod.notmiwae.value):
            output_path = (pathlib.Path(__file__).parent.parent.parent
                           .joinpath('results')
                           .joinpath('intermediate_state')
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
        self._logger.info('Nulls are successfully imputed')

        return X_train_imputed, X_tests_imputed_lst, null_imputer_params_dct, imputation_runtime

    def _evaluate_imputation(self, real, imputed, corrupted, numerical_columns, null_imputer_name, null_imputer_params_dct):
        group_indexes_dct = create_test_protected_groups(real, real, self.virny_config.sensitive_attributes_dct)
        overall_grp = 'overall'
        subgroups = [overall_grp] + list(group_indexes_dct.keys())
        columns_with_nulls = corrupted.columns[corrupted.isna().any()].tolist()
        metrics_df = pd.DataFrame(columns=('Dataset_Name', 'Null_Imputer_Name', 'Null_Imputer_Params',
                                           'Column_Type', 'Column_With_Nulls', 'Subgroup', 'Sample_Size',
                                           'KL_Divergence_Pred', 'KL_Divergence_Total',
                                           'RMSE', 'Precision', 'Recall', 'F1_Score'))
        for column_idx, column_name in enumerate(columns_with_nulls):
            column_type = 'numerical' if column_name in numerical_columns else 'categorical'

            for subgroup_idx, subgroup_name in enumerate(subgroups):
                verbose = True if subgroup_name == overall_grp else False
                indexes = corrupted[column_name].isna() if subgroup_name == overall_grp \
                    else corrupted[corrupted[column_name].isna()].index.intersection(group_indexes_dct[subgroup_name].index)
                if len(indexes) == 0:
                    print(f'Nulls were not injected to any {subgroup_name} row. Skipping...')
                    continue

                true = real.loc[indexes, column_name]
                pred = imputed.loc[indexes, column_name]
                if subgroup_name == overall_grp:
                    grp_total_real = real[column_name]
                    grp_total_imputed = imputed[column_name]
                else:
                    grp_total_real = real.loc[group_indexes_dct[subgroup_name].index, column_name]
                    grp_total_imputed = imputed.loc[group_indexes_dct[subgroup_name].index, column_name]

                # If an initial dataset contains realistic nulls, do not include them
                # in the imputation performance measurement
                if real[column_name].isnull().sum() > 0:
                    if verbose:
                        print('WARNING: an initial dataset includes existing realistic nulls')

                    true = true[~true.isnull()]
                    pred = pred.loc[true.index]
                    grp_total_real = grp_total_real[~grp_total_real.isnull()]
                    grp_total_imputed = grp_total_imputed.loc[grp_total_real.index]

                # Column type agnostic metrics
                kl_divergence_pred = calculate_kl_divergence(true=true,
                                                             pred=pred,
                                                             column_type=column_type,
                                                             verbose=verbose)
                kl_divergence_total = calculate_kl_divergence(true=grp_total_real,
                                                              pred=grp_total_imputed,
                                                              column_type=column_type,
                                                              verbose=verbose)
                if verbose:
                    if kl_divergence_pred: 
                        print('Predictive KL divergence for {}: {:.2f}'.format(column_name, kl_divergence_pred))
                    else:
                        print('Predictive KL divergence for {}: None'.format(column_name))
                    if kl_divergence_total:
                        print('Total KL divergence for {}: {:.2f}'.format(column_name, kl_divergence_total))
                    else:
                        print('Total KL divergence for {}: None'.format(column_name))

                rmse = None
                precision = None
                recall = None
                f1 = None
                if column_type == 'numerical':
                    null_imputer_params = null_imputer_params_dct[column_name] if null_imputer_params_dct is not None else None

                    # Scale numerical features before computing RMSE
                    scaler = StandardScaler()
                    true_scaled = scaler.fit_transform(true.to_frame()).flatten().astype(float)
                    pred_scaled = scaler.transform(pred.to_frame()).flatten().astype(float)

                    rmse = mean_squared_error(true_scaled, pred_scaled, squared=False)
                    if verbose:
                        print('RMSE for {}: {:.2f}'.format(column_name, rmse))
                else:
                    null_imputer_params = null_imputer_params_dct[column_name] if null_imputer_params_dct is not None else None
                    precision, recall, f1, _ = precision_recall_fscore_support(true, pred, average="micro")
                    if verbose:
                        print('Precision for {}: {:.2f}'.format(column_name, precision))
                        print('Recall for {}: {:.2f}'.format(column_name, recall))
                        print('F1 score for {}: {:.2f}'.format(column_name, f1))

                if verbose:
                    print('\n')

                # Save imputation performance metric of the imputer in a dataframe
                new_row_idx = column_idx * len(subgroups) + subgroup_idx
                metrics_df.loc[new_row_idx] = [self.dataset_name, null_imputer_name, null_imputer_params,
                                               column_type, column_name, subgroup_name, true.shape[0],
                                               kl_divergence_pred, kl_divergence_total,
                                               rmse, precision, recall, f1]

        return metrics_df

    def _save_imputation_metrics_to_db(self, train_imputation_metrics_df: pd.DataFrame, test_imputation_metrics_dfs_lst: list,
                                       imputation_runtime: float, null_imputer_name: str, evaluation_scenario: str, 
                                       experiment_seed: int, null_imputer_params_dct: dict):
        train_imputation_metrics_df['Imputation_Guid'] = train_imputation_metrics_df.apply(
            lambda row: generate_guid(ordered_hierarchy_lst=[self.dataset_name, null_imputer_name,
                                                             evaluation_scenario, experiment_seed,
                                                             'X_train_val', row['Column_With_Nulls'],
                                                             row['Subgroup']]),
            axis=1
        )
        self._db.write_pandas_df_into_db(collection_name=IMPUTATION_PERFORMANCE_METRICS_COLLECTION_NAME,
                                         df=train_imputation_metrics_df,
                                         custom_tbl_fields_dct={
                                             'exp_pipeline_guid': generate_guid(ordered_hierarchy_lst=[self.dataset_name, null_imputer_name, evaluation_scenario, experiment_seed]),
                                             'session_uuid': self._session_uuid,
                                             'evaluation_scenario': evaluation_scenario,
                                             'experiment_seed': experiment_seed,
                                             'dataset_part': 'X_train_val',
                                             'runtime_in_mins': imputation_runtime,
                                             'record_create_date_time': datetime.now(timezone.utc),
                                             'null_imputer_params_dct': null_imputer_params_dct
                                         })

        # Save imputation results into a database for each test set from the evaluation scenario
        _, test_injection_scenarios_lst = get_injection_scenarios(evaluation_scenario)
        test_record_create_date_time = datetime.now(timezone.utc)
        for test_set_idx, test_imputation_metrics_df in enumerate(test_imputation_metrics_dfs_lst):
            test_injection_scenario = test_injection_scenarios_lst[test_set_idx]
            test_imputation_metrics_df['Imputation_Guid'] = test_imputation_metrics_df.apply(
                lambda row: generate_guid(ordered_hierarchy_lst=[self.dataset_name, null_imputer_name,
                                                                 evaluation_scenario, experiment_seed,
                                                                 f'X_test_{test_injection_scenario}',
                                                                 row['Column_With_Nulls'], row['Subgroup']]),
                axis=1
            )
            self._db.write_pandas_df_into_db(collection_name=IMPUTATION_PERFORMANCE_METRICS_COLLECTION_NAME,
                                             df=test_imputation_metrics_df,
                                             custom_tbl_fields_dct={
                                                 'exp_pipeline_guid': generate_guid(ordered_hierarchy_lst=[self.dataset_name, null_imputer_name, evaluation_scenario, experiment_seed]),
                                                 'session_uuid': self._session_uuid,
                                                 'evaluation_scenario': evaluation_scenario,
                                                 'experiment_seed': experiment_seed,
                                                 'dataset_part': f'X_test_{test_injection_scenario}',
                                                 'runtime_in_mins': imputation_runtime,
                                                 'record_create_date_time': test_record_create_date_time,
                                                 'null_imputer_params_dct': null_imputer_params_dct
                                            })

        self._logger.info("Performance metrics and tuned parameters of the null imputer are saved into a database")

    def _save_imputed_datasets_to_fs(self, X_train_val: pd.DataFrame, X_tests_lst: pd.DataFrame,
                                     null_imputer_name: str, evaluation_scenario: str, experiment_seed: int):
        save_sets_dir_path = (pathlib.Path(__file__).parent.parent.parent
                              .joinpath('results')
                              .joinpath('imputed_datasets')
                              .joinpath(self.dataset_name)
                              .joinpath(null_imputer_name)
                              .joinpath(evaluation_scenario)
                              .joinpath(str(experiment_seed)))
        os.makedirs(save_sets_dir_path, exist_ok=True)

        train_set_filename = f'imputed_{self.dataset_name}_{null_imputer_name}_{evaluation_scenario}_{experiment_seed}_X_train_val.csv'
        X_train_val.to_csv(os.path.join(save_sets_dir_path, train_set_filename),
                           sep=",",
                           columns=X_train_val.columns,
                           index=True)

        # Save each imputed test set in a local filesystem
        _, test_injection_scenarios_lst = get_injection_scenarios(evaluation_scenario)
        for test_set_idx, X_test in enumerate(X_tests_lst):
            test_injection_scenario = test_injection_scenarios_lst[test_set_idx]
            test_set_filename = f'imputed_{self.dataset_name}_{null_imputer_name}_{evaluation_scenario}_{experiment_seed}_X_test_{test_injection_scenario}.csv'
            X_test.to_csv(os.path.join(save_sets_dir_path, test_set_filename),
                          sep=",",
                          columns=X_test.columns,
                          index=True)

        self._logger.info("Imputed train and test sets are saved locally")

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
