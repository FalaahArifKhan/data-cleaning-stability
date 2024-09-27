"""
Script for evaluating model performance on the imputed datasets from impute_nulls_with_predictor.py
"""
import os
import sys
import warnings
from pathlib import Path

# Suppress all warnings
warnings.filterwarnings("ignore")

# Define a correct root path
sys.path.append(str(Path(f"{__file__}").parent.parent))

# Import dependencies
import argparse
from datetime import datetime
from dotenv import load_dotenv

from configs.scenarios_config import EVALUATION_SCENARIOS_CONFIG
from source.custom_classes.benchmark import Benchmark
from source.custom_classes.database_client import get_secrets_path
from source.validation import validate_args, str2bool


def preconfigure_experiment(env_file_path: str = Path(__file__).parent.joinpath('..', 'configs', 'secrets.env')):
    os.environ["PYTHONWARNINGS"] = "ignore"

    # Load env variables
    load_dotenv(env_file_path)
    print('\n\nDB_NAME secret:', os.getenv("DB_NAME"))


def parse_input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="a dataset name", required=True)
    parser.add_argument("--null_imputers", type=str, help="a list of null imputers", required=True)
    parser.add_argument("--models", type=str, help="a list of model names", required=True)
    parser.add_argument("--run_nums", type=str, help="a list of experiment run numbers", required=True)
    parser.add_argument("--tune_imputers", type=str2bool, required=True,
                        help="True -- tune null imputers, False -- take hyper-params of null imputers from configs/null_imputers_config.py")
    parser.add_argument("--save_imputed_datasets", type=str2bool, required=False, default=False,
                        help="True -- save imputed train and test sets, False -- do not save train and test sets")
    parser.add_argument("--ml_impute", type=str2bool, required=True,
                        help="True -- apply ML-oriented imputers, False -- use pre-computed imputed datasets")
    parser.add_argument("--evaluation_scenarios", type=str, help="a list of evaluation scenarios",
                        default=str(list(EVALUATION_SCENARIOS_CONFIG.keys())))

    args = parser.parse_args()
    args = validate_args(exp_config_obj=args)

    print(
        f"Dataset name: {args.dataset}\n"
        f"Null imputers: {args.null_imputers}\n"
        f"Models: {args.models}\n"
        f"Run nums: {args.run_nums}\n"
        f"Evaluation scenarios: {args.evaluation_scenarios}\n"
    )

    return args


if __name__ == '__main__':
    start_time = datetime.now()
    preconfigure_experiment(env_file_path=get_secrets_path('secrets.env'))
    args = parse_input_args()

    benchmark = Benchmark(dataset_name=args.dataset,
                          null_imputers=args.null_imputers,
                          model_names=args.models)
    benchmark.run_experiment(run_nums=args.run_nums,
                             evaluation_scenarios=args.evaluation_scenarios,
                             model_names=args.models,
                             tune_imputers=args.tune_imputers,
                             ml_impute=args.ml_impute,
                             save_imputed_datasets=args.save_imputed_datasets)

    end_time = datetime.now()
    print(f'The script is successfully executed. Run time: {end_time - start_time}')
    print('Session UUID for all results:', benchmark._session_uuid)
