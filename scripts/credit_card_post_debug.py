import os
import sys
import warnings
warnings.filterwarnings('ignore')
os.environ["PYTHONWARNINGS"] = "ignore"

sys.path.append(r"c:\RAI\fairness-variance-latest\fairness-variance")
print(os.getcwd())

import os
import copy

from virny.utils.custom_initializers import create_config_obj
from virny.datasets import CreditCardDefaultDataset

from configs.constants import TEST_SET_FRACTION, EXPERIMENT_SEEDS
from configs.models_config_for_tuning import get_model_params_for_postprocessing
from source.experiment_interface import run_exp_iter_with_eqq_odds_postprocessing_debug


ROOT_DIR = os.getcwd()
EXPERIMENT_NAME = 'eq_odds_postprocessing_credit_card'
DB_COLLECTION_NAME = 'eq_odds_postprocessing'
SAVE_RESULTS_DIR_PATH = os.path.join(ROOT_DIR, 'results', EXPERIMENT_NAME)

config_yaml_path = r"C:\RAI\fairness-variance-latest\fairness-variance\notebooks\eq_odds_postprocessing_credit_card\credit_card_default_config.yaml"
#os.path.join(ROOT_DIR, 'notebooks', EXPERIMENT_NAME, 'credit_card_default_config.yaml')
metrics_computation_config = create_config_obj(config_yaml_path=config_yaml_path)

import os
from dotenv import load_dotenv

load_dotenv('./configs/secrets.env')
os.getenv("DB_NAME")

from source.utils.db_functions import connect_to_mongodb

client, collection_obj, db_writer_func = connect_to_mongodb(DB_COLLECTION_NAME)

import uuid

custom_table_fields_dct = {
    #'session_uuid': str(uuid.uuid4()),
    'session_uuid': "test_session_debug_postprocessing"
}
print('Current session uuid: ', custom_table_fields_dct['session_uuid'])

data_loader = CreditCardDefaultDataset()

tuned_params_filenames = [
    'tuning_results_Credit_Card_Default_20231023__200422.csv'
]
tuned_params_df_paths = [r"C:\RAI\fairness-variance-latest\fairness-variance\results\eq_odds_postprocessing_credit_card\tuning_results_Credit_Card_Default_20231023__200422.csv"]
#[os.path.join(ROOT_DIR, 'results', EXPERIMENT_NAME, tuned_params_filename)
 #                        for tuned_params_filename in tuned_params_filenames]

# Configs for an experiment iteration
exp_iter_num = 1
experiment_seed = EXPERIMENT_SEEDS[exp_iter_num - 1]
custom_table_fields_dct['experiment_iteration'] = f'Exp_iter_{exp_iter_num}'

exp_iter_data_loader = copy.deepcopy(data_loader)  # Add deepcopy to avoid data leakage
models_params_for_tuning = get_model_params_for_postprocessing(experiment_seed)

run_exp_iter_with_eqq_odds_postprocessing_debug(data_loader=exp_iter_data_loader,
                                           experiment_seed=experiment_seed,
                                           dataset_split_seed=experiment_seed,
                                           test_set_fraction=TEST_SET_FRACTION,
                                           db_writer_func=db_writer_func,
                                           models_params_for_tuning=models_params_for_tuning,
                                           metrics_computation_config=metrics_computation_config,
                                           custom_table_fields_dct=custom_table_fields_dct,
                                           with_tuning=False,
                                           tuned_params_df_path=tuned_params_df_paths[0],
                                           save_results_dir_path=SAVE_RESULTS_DIR_PATH,
                                           verbose=True,
                                           dataset_name='CreditCardDefaultDataset')