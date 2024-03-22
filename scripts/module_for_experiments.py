import os
import sys
from pathlib import Path

# Define a correct root path
sys.path.append(str(Path(f"{__file__}").parent.parent.parent))
print('Current location: ', os.getcwd())

# Import dependencies
import ast
import copy
import argparse
import warnings
from datetime import datetime
from dotenv import load_dotenv
from IPython.display import display

from virny.utils.custom_initializers import create_config_obj
from virny.datasets import ACSIncomeDataset

from configs.constants import TEST_SET_FRACTION, EXPERIMENT_SEEDS
from configs.models_config_for_tuning import get_folktables_employment_models_params_for_tuning
from source.utils.db_functions import connect_to_mongodb
from source.experiment_interface import run_exp_iter_with_disparate_impact_and_mult_sets


# Define configurable variables
EXPERIMENT_NAME = 'out_of_domain_uncertainty_rich_income_LFR'
DB_COLLECTION_NAME = 'out_of_domain_uncertainty'
TRAIN_SET_SUBSAMPLE_SIZE = 5_000
CUSTOM_TABLE_FIELDS_DCT = {
    'session_uuid': '0c3ad11b-5085-478a-b3ae-f5fdecbfca77',  # str(uuid.uuid4())
}

# Define input variables
ROOT_DIR = os.getcwd()
SAVE_RESULTS_DIR_PATH = os.path.join(ROOT_DIR, 'results', EXPERIMENT_NAME, os.path.basename(__file__))
config_yaml_path = os.path.join(ROOT_DIR, 'notebooks', EXPERIMENT_NAME, 'rich_income_2018_config.yaml')
METRICS_COMPUTATION_CONFIG = create_config_obj(config_yaml_path=config_yaml_path)
CLIENT, COLLECTION_OBJ, DB_WRITER_FUNC = connect_to_mongodb(DB_COLLECTION_NAME)


def preconfigurate_experiment(env_file_path='./configs/secrets.env'):
    warnings.filterwarnings('ignore')
    os.environ["PYTHONWARNINGS"] = "ignore"

    # Load env variables
    load_dotenv(env_file_path)
    print('\n\nDB_NAME:', os.getenv("DB_NAME"))


def parse_input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_nums", type=str, help="a list of experiment run numbers", required=True)
    parser.add_argument("--fairness_intervention_params", type=str,
                        help="a list of fairness intervention params", required=True)
    parser.add_argument("--tuned_params_filenames", type=str,
                        help="a list of filenames with tuned model hyper-parameters", default='[]')
    args = parser.parse_args()

    run_nums = ast.literal_eval(args.run_nums)
    fairness_intervention_params = ast.literal_eval(args.fairness_intervention_params)
    tuned_params_filenames = ast.literal_eval(args.tuned_params_filenames)

    print(
        f"Experiment name: {EXPERIMENT_NAME}\n"
        f"Current session uuid: {CUSTOM_TABLE_FIELDS_DCT['session_uuid']}\n"
        f"Experiment run numbers: {run_nums}\n"
        f"Fairness intervention params: {fairness_intervention_params}\n"
        f"Train set size: {TRAIN_SET_SUBSAMPLE_SIZE}\n"
        f"Tuned params filenames: {tuned_params_filenames}\n"
    )

    return run_nums, fairness_intervention_params, tuned_params_filenames


def run_experiment(exp_run_num, fairness_intervention_params, tuned_params_filenames,
                   data_loader_rich, extra_data_loaders):
    # Configs for an experiment iteration
    experiment_seed = EXPERIMENT_SEEDS[exp_run_num - 1]
    CUSTOM_TABLE_FIELDS_DCT['experiment_iteration'] = f'Exp_iter_{exp_run_num}'

    exp_iter_data_loader = copy.deepcopy(data_loader_rich)  # Add deepcopy to avoid data leakage
    exp_extra_data_loaders = copy.deepcopy(extra_data_loaders)  # Add deepcopy to avoid data leakage
    models_params_for_tuning = get_folktables_employment_models_params_for_tuning(experiment_seed)

    if exp_run_num == 1:
        with_tuning = True
        tuned_params_df_paths = None
        print('Enable hyper-params tuning')
    else:
        with_tuning = False
        tuned_params_df_paths = [os.path.join(ROOT_DIR, 'results', EXPERIMENT_NAME, os.path.basename(__file__), tuned_params_filename)
                                 for tuned_params_filename in tuned_params_filenames]

    run_exp_iter_with_disparate_impact_and_mult_sets(data_loader=exp_iter_data_loader,
                                                     extra_data_loaders=exp_extra_data_loaders,
                                                     experiment_seed=experiment_seed,
                                                     test_set_fraction=TEST_SET_FRACTION,
                                                     db_writer_func=DB_WRITER_FUNC,
                                                     fair_intervention_params_lst=fairness_intervention_params,
                                                     models_params_for_tuning=models_params_for_tuning,
                                                     metrics_computation_config=METRICS_COMPUTATION_CONFIG,
                                                     custom_table_fields_dct=CUSTOM_TABLE_FIELDS_DCT,
                                                     with_tuning=with_tuning,
                                                     tuned_params_df_paths=tuned_params_df_paths,
                                                     save_results_dir_path=SAVE_RESULTS_DIR_PATH,
                                                     train_set_subsample_size=TRAIN_SET_SUBSAMPLE_SIZE,
                                                     verbose=True)


if __name__ == '__main__':
    start_time = datetime.now()
    preconfigurate_experiment()
    run_nums, fairness_intervention_params, tuned_params_filenames = parse_input_args()

    # Initialize custom objects
    data_loader_rich = ACSIncomeDataset(state=['MD', 'NJ', 'MA'], year=2018, with_nulls=False,
                                        subsample_size=100_000, subsample_seed=42)
    print('data_loader_rich.X_data.shape:', data_loader_rich.X_data.shape)
    print('data_loader_rich.X_data.head()')
    display(data_loader_rich.X_data.head())

    data_loader_poor = ACSIncomeDataset(state=['WV', 'MS', 'AR', 'NM', 'LA', 'AL', 'KY'], year=2018, with_nulls=False,
                                        subsample_size=100_000, subsample_seed=42)
    print('data_loader_poor.X_data.shape:', data_loader_poor.X_data.shape)
    print('data_loader_poor.X_data.head()')
    display(data_loader_poor.X_data.head())

    extra_data_loaders = [data_loader_poor]

    # Execute each experiment run
    for run_num in run_nums:
        run_experiment(exp_run_num=run_num,
                       fairness_intervention_params=fairness_intervention_params,
                       tuned_params_filenames=tuned_params_filenames,
                       data_loader_rich=data_loader_rich,
                       extra_data_loaders=extra_data_loaders)
        print('\n\n\n', flush=True)

    end_time = datetime.now()
    print(f'The script is successfully executed. Run time: {end_time - start_time}')

    CLIENT.close()
