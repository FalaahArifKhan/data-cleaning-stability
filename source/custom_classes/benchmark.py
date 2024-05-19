import os
import copy
import tqdm
import pathlib
import pandas as pd
from pprint import pprint

from sklearn.model_selection import train_test_split

from virny.user_interfaces.multiple_models_with_db_writer_api import compute_metrics_with_db_writer
from virny.user_interfaces.multiple_models_with_multiple_test_sets_api import compute_metrics_with_multiple_test_sets

from configs.null_imputers_config import NULL_IMPUTERS_CONFIG
from configs.constants import (EXP_COLLECTION_NAME, EXPERIMENT_RUN_SEEDS, ErrorRepairMethod)
from source.utils.common_helpers import (generate_guid, create_virny_base_flow_datasets, get_injection_scenarios)
from source.utils.dataframe_utils import preprocess_base_flow_dataset, preprocess_mult_base_flow_datasets
from source.custom_classes.ml_lifecycle import MLLifecycle


class Benchmark(MLLifecycle):
    """
    Class encapsulates all experimental pipelines
    """
    def __init__(self, dataset_name: str, null_imputers: list, model_names: list):
        """
        Constructor defining default variables
        """
        super().__init__(dataset_name=dataset_name,
                         null_imputers=null_imputers,
                         model_names=model_names)

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

        self._logger.info("Start an experiment iteration for the following custom params:")
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
        self.virny_config.random_state = experiment_seed  # Set random state for the metric computation with Virny
        compute_metrics_with_db_writer(dataset=base_flow_dataset,
                                       config=self.virny_config,
                                       models_config=models_config,
                                       custom_tbl_fields_dct=custom_table_fields_dct,
                                       db_writer_func=self._db.get_db_writer(collection_name=EXP_COLLECTION_NAME),
                                       notebook_logs_stdout=False,
                                       verbose=0)

    def evaluate_baselines(self, run_nums: list, model_names: list):
        self._db.connect()

        total_iterations = len(run_nums)
        with tqdm.tqdm(total=total_iterations, desc="Baseline Evaluation Progress") as pbar:
            for run_idx, run_num in enumerate(run_nums):
                self._logger.info(f"{'=' * 30} NEW BASELINE EVALUATION RUN {'=' * 30}")
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

        self._db.close()
        self._logger.info("Performance metrics of the baselines were successfully saved!")

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
            self._logger.info('Evaluating imputation for X_train_val...')
            train_imputation_metrics_df = self._evaluate_imputation(real=X_train_val,
                                                                    corrupted=X_train_val_with_nulls_wo_sensitive_attrs,
                                                                    imputed=X_train_val_imputed_wo_sensitive_attrs,
                                                                    numerical_columns=numerical_columns_wo_sensitive_attrs,
                                                                    null_imputer_name=null_imputer_name,
                                                                    null_imputer_params_dct=null_imputer_params_dct)
        print('\n')
        self._logger.info('Evaluating imputation for X_test sets...')
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
                                            experiment_seed=experiment_seed,
                                            null_imputer_params_dct=null_imputer_params_dct)

        if save_imputed_datasets:
            self._save_imputed_datasets_to_fs(X_train_val=X_train_val_imputed_wo_sensitive_attrs,
                                              X_tests_lst=X_tests_imputed_wo_sensitive_attrs_lst,
                                              null_imputer_name=null_imputer_name,
                                              evaluation_scenario=evaluation_scenario,
                                              experiment_seed=experiment_seed)

        # Create base flow datasets for Virny to compute metrics
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
        self._db.connect()

        total_iterations = len(self.null_imputers) * len(evaluation_scenarios) * len(run_nums)
        with tqdm.tqdm(total=total_iterations, desc="Null Imputation Progress") as pbar:
            for null_imputer_idx, null_imputer_name in enumerate(self.null_imputers):
                for evaluation_scenario_idx, evaluation_scenario in enumerate(evaluation_scenarios):
                    for run_idx, run_num in enumerate(run_nums):
                        self._logger.info(f"{'=' * 30} NEW DATASET IMPUTATION RUN {'=' * 30}")
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

        self._db.close()
        self._logger.info("Experimental results were successfully saved!")

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
                                  .joinpath('imputed_datasets')
                                  .joinpath(self.dataset_name)
                                  .joinpath(null_imputer_name)
                                  .joinpath(evaluation_scenario)
                                  .joinpath(str(experiment_seed)))

        # Create a base flow dataset for Virny to compute metrics
        numerical_columns_wo_sensitive_attrs = [col for col in data_loader.numerical_columns if col not in self.dataset_sensitive_attrs]
        categorical_columns_wo_sensitive_attrs = [col for col in data_loader.categorical_columns if col not in self.dataset_sensitive_attrs]

        # Read X_train_val set
        train_set_filename = f'imputed_{self.dataset_name}_{null_imputer_name}_{evaluation_scenario}_{experiment_seed}_X_train_val.csv'
        X_train_val_imputed_wo_sensitive_attrs = pd.read_csv(os.path.join(save_sets_dir_path, train_set_filename),
                                                             header=0, index_col=0)
        X_train_val_imputed_wo_sensitive_attrs[categorical_columns_wo_sensitive_attrs] = (
            X_train_val_imputed_wo_sensitive_attrs[categorical_columns_wo_sensitive_attrs].astype(str))

        # Subset y_train_val to align with X_train_val_imputed_wo_sensitive_attrs
        if null_imputer_name == ErrorRepairMethod.deletion.value:
            y_train_val = y_train_val.loc[X_train_val_imputed_wo_sensitive_attrs.index]

        # Read X_test sets
        X_tests_imputed_wo_sensitive_attrs_lst = list()
        _, test_injection_scenarios_lst = get_injection_scenarios(evaluation_scenario)
        for test_injection_scenario in test_injection_scenarios_lst:
            test_set_filename = f'imputed_{self.dataset_name}_{null_imputer_name}_{evaluation_scenario}_{experiment_seed}_X_test_{test_injection_scenario}.csv'
            X_test_imputed_wo_sensitive_attrs = pd.read_csv(os.path.join(save_sets_dir_path, test_set_filename),
                                                            header=0, index_col=0)
            X_test_imputed_wo_sensitive_attrs[categorical_columns_wo_sensitive_attrs] = (
                X_test_imputed_wo_sensitive_attrs[categorical_columns_wo_sensitive_attrs].astype(str))
            X_tests_imputed_wo_sensitive_attrs_lst.append(X_test_imputed_wo_sensitive_attrs)

        # Create base flow datasets for Virny to compute metrics
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

    def _run_exp_iter_for_joint_cleaning_and_training(self, data_loader, experiment_seed: int, evaluation_scenario: str,
                                                      null_imputer_name: str, tune_imputers: bool,
                                                      model_names: list, custom_table_fields_dct: dict):
        if len(model_names) > 0 and null_imputer_name == ErrorRepairMethod.cp_clean.value:
            self._logger.warning(f'model_names argument is ignored for {ErrorRepairMethod.cp_clean.value} '
                                 f'since only KNN is supported for this null imputation method')

        # Split the dataset
        X_train_val, X_test, y_train_val, y_test = self._split_dataset(data_loader, experiment_seed)

        # Inject nulls not into sensitive attributes
        X_train_val_with_nulls, X_tests_with_nulls_lst = self._inject_nulls(X_train_val=X_train_val,
                                                                            X_test=X_test,
                                                                            evaluation_scenario=evaluation_scenario,
                                                                            experiment_seed=experiment_seed)

        # Remove sensitive attributes from train and test sets to avoid their usage during imputation
        (X_train_val_wo_sensitive_attrs,
         X_test_wo_sensitive_attrs_lst, _, _) = self._remove_sensitive_attrs(X_train_val=X_train_val,
                                                                             X_tests_lst=[X_test],
                                                                             data_loader=data_loader)
        (X_train_val_with_nulls_wo_sensitive_attrs,
         X_tests_with_nulls_wo_sensitive_attrs_lst,
         numerical_columns_wo_sensitive_attrs,
         categorical_columns_wo_sensitive_attrs) = self._remove_sensitive_attrs(X_train_val=X_train_val_with_nulls,
                                                                                X_tests_lst=X_tests_with_nulls_lst,
                                                                                data_loader=data_loader)

        # Define a directory path to save an intermediate state
        save_dir = (pathlib.Path(__file__).parent.parent.parent
                        .joinpath('results')
                        .joinpath('intermediate_state')
                        .joinpath(null_imputer_name)
                        .joinpath(self.dataset_name)
                        .joinpath(evaluation_scenario)
                        .joinpath(str(experiment_seed)))

        # Use a method, kwargs, and hyperparams from NULL_IMPUTERS_CONFIG
        joint_cleaning_and_training_func = NULL_IMPUTERS_CONFIG[null_imputer_name]["method"]
        imputation_kwargs = NULL_IMPUTERS_CONFIG[null_imputer_name]["kwargs"]
        imputation_kwargs.update({'save_dir': save_dir})       
        imputation_kwargs['tune'] = tune_imputers
        print("imputation_kwargs['tune']", imputation_kwargs['tune'])
        
        # Make paths for the imputed datasets
        imputed_datasets_paths = []
        results_dir = pathlib.Path(__file__).parent.parent.parent.joinpath('results').joinpath('imputed_datasets')
        
        for dateset_dir in results_dir.iterdir():
            if dateset_dir.is_dir() and dateset_dir.name == self.dataset_name:
                for method_dir in dateset_dir.iterdir():
                    if method_dir.name == 'deletion':
                        continue
                    for scenario_dir in method_dir.iterdir():
                        if scenario_dir.name == evaluation_scenario:
                            for seed_dir in scenario_dir.iterdir():
                                if seed_dir.name == str(experiment_seed):
                                    for file in seed_dir.iterdir():
                                        if file.is_file() and file.name.startswith('imputed') and 'X_train' in file.name:
                                            print('Used imputed dataset', file)
                                            imputed_datasets_paths.append(file)
        
        imputation_kwargs['computed_repaired_datasets_paths'] = imputed_datasets_paths if len(imputed_datasets_paths) > 0 else None                                

        # Create a wrapper for the input joint cleaning-and-training method
        # to conduct in-depth performance profiling with Virny
        (models_config,
         X_train_with_nulls_wo_sensitive_attrs,
         y_train) = joint_cleaning_and_training_func(X_train_val=X_train_val_wo_sensitive_attrs,
                                                     y_train_val=y_train_val,
                                                     X_train_val_with_nulls=X_train_val_with_nulls_wo_sensitive_attrs,
                                                     numerical_columns=numerical_columns_wo_sensitive_attrs,
                                                     categorical_columns=categorical_columns_wo_sensitive_attrs,
                                                     experiment_seed=experiment_seed,
                                                     **imputation_kwargs)

        # Create a base flow dataset for Virny to compute metrics
        main_base_flow_dataset, extra_base_flow_datasets = \
            create_virny_base_flow_datasets(data_loader=data_loader,
                                            dataset_sensitive_attrs=self.dataset_sensitive_attrs,
                                            X_train_val_wo_sensitive_attrs=X_train_with_nulls_wo_sensitive_attrs,
                                            X_tests_wo_sensitive_attrs_lst=X_tests_with_nulls_wo_sensitive_attrs_lst,
                                            y_train_val=y_train,
                                            y_test=y_test,
                                            numerical_columns_wo_sensitive_attrs=numerical_columns_wo_sensitive_attrs,
                                            categorical_columns_wo_sensitive_attrs=categorical_columns_wo_sensitive_attrs)

        # Create extra_test_sets
        extra_test_sets = [(extra_base_flow_datasets[i].X_test, extra_base_flow_datasets[i].y_test, extra_base_flow_datasets[i].init_sensitive_attrs_df)
                           for i in range(len(extra_base_flow_datasets))]

        # Compute metrics using Virny
        self.virny_config.random_state = experiment_seed  # Set random state for the metric computation with Virny
        compute_metrics_with_multiple_test_sets(dataset=main_base_flow_dataset,
                                                extra_test_sets_lst=extra_test_sets,
                                                config=self.virny_config,
                                                models_config=models_config,
                                                custom_tbl_fields_dct=custom_table_fields_dct,
                                                db_writer_func=self._db.get_db_writer(collection_name=EXP_COLLECTION_NAME),
                                                with_predict_proba=False,  # joint cleaning-and-training models do not support a predict_proba method
                                                notebook_logs_stdout=False,
                                                verbose=0)

    def _run_exp_iter_for_standard_imputation(self, data_loader, experiment_seed: int, evaluation_scenario: str,
                                              null_imputer_name: str, model_names: list, tune_imputers: bool,
                                              ml_impute: bool, save_imputed_datasets: bool, custom_table_fields_dct: dict):
        if ml_impute:
            main_base_flow_dataset, extra_base_flow_datasets = self.inject_and_impute_nulls(data_loader=data_loader,
                                                                                            null_imputer_name=null_imputer_name,
                                                                                            evaluation_scenario=evaluation_scenario,
                                                                                            tune_imputers=tune_imputers,
                                                                                            experiment_seed=experiment_seed,
                                                                                            save_imputed_datasets=save_imputed_datasets)
        else:
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
        self.virny_config.random_state = experiment_seed  # Set random state for the metric computation with Virny
        compute_metrics_with_multiple_test_sets(dataset=main_base_flow_dataset,
                                                extra_test_sets_lst=extra_test_sets,
                                                config=self.virny_config,
                                                models_config=models_config,
                                                custom_tbl_fields_dct=custom_table_fields_dct,
                                                db_writer_func=self._db.get_db_writer(collection_name=EXP_COLLECTION_NAME),
                                                notebook_logs_stdout=False,
                                                verbose=0)

    def _run_exp_iter(self, init_data_loader, run_num, evaluation_scenario, null_imputer_name,
                      model_names, tune_imputers, ml_impute, save_imputed_datasets):
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

        self._logger.info("Start an experiment iteration for the following custom params:")
        pprint(custom_table_fields_dct)
        print('\n', flush=True)

        if null_imputer_name in (ErrorRepairMethod.boost_clean.value, ErrorRepairMethod.cp_clean.value):
            self._run_exp_iter_for_joint_cleaning_and_training(data_loader=data_loader,
                                                               experiment_seed=experiment_seed,
                                                               evaluation_scenario=evaluation_scenario,
                                                               null_imputer_name=null_imputer_name,
                                                               model_names=model_names,
                                                               tune_imputers=tune_imputers,
                                                               custom_table_fields_dct=custom_table_fields_dct)
        else:
            self._run_exp_iter_for_standard_imputation(data_loader=data_loader,
                                                       experiment_seed=experiment_seed,
                                                       evaluation_scenario=evaluation_scenario,
                                                       null_imputer_name=null_imputer_name,
                                                       model_names=model_names,
                                                       tune_imputers=tune_imputers,
                                                       ml_impute=ml_impute,
                                                       save_imputed_datasets=save_imputed_datasets,
                                                       custom_table_fields_dct=custom_table_fields_dct)

    def run_experiment(self, run_nums: list, evaluation_scenarios: list, model_names: list,
                       tune_imputers: bool, ml_impute: bool, save_imputed_datasets: bool):
        self._db.connect()

        total_iterations = len(self.null_imputers) * len(evaluation_scenarios) * len(run_nums)
        with tqdm.tqdm(total=total_iterations, desc="Experiment Progress") as pbar:
            for null_imputer_idx, null_imputer_name in enumerate(self.null_imputers):
                for evaluation_scenario_idx, evaluation_scenario in enumerate(evaluation_scenarios):
                    for run_idx, run_num in enumerate(run_nums):
                        self._logger.info(f"{'=' * 30} NEW EXPERIMENT RUN {'=' * 30}")
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
                                           ml_impute=ml_impute,
                                           save_imputed_datasets=save_imputed_datasets)
                        pbar.update(1)
                        print('\n\n\n\n', flush=True)

        self._db.close()
        self._logger.info("Experimental results were successfully saved!")
