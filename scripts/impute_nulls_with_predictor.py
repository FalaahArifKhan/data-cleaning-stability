"""
Script for repairing various errors in the datasets
"""
import os
import sys
import warnings
from pathlib import Path

# Remove warnings
warnings.filterwarnings('ignore')
os.environ["PYTHONWARNINGS"] = "ignore"

# Define a correct root path
sys.path.append(str(Path(__file__).parent.parent))
print('Current location: ', os.getcwd())

# Import dependencies
import argparse
from datetime import datetime
from dotenv import load_dotenv

from configs.constants import EVALUATION_SCENARIOS
from source.custom_classes.benchmark import Benchmark
from source.validation import validate_args, str2bool


def preconfigure_experiment(env_file_path: str = Path(__file__).parent.joinpath('..', 'configs', 'secrets.env')):
    # Load env variables
    load_dotenv(env_file_path)
    print('\n\nDB_NAME secret:', os.getenv("DB_NAME"))


def parse_input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="a dataset name", required=True)
    parser.add_argument("--null_imputers", type=str, help="a list of null imputers", required=True)
    parser.add_argument("--run_nums", type=str, help="a list of experiment run numbers", required=True)
    parser.add_argument("--tune_imputers", type=str2bool, required=True,
                        help="True -- tune null imputers, False -- take hyper-params of null imputers from configs/null_imputers_config.py")
    parser.add_argument("--save_imputed_datasets", type=str2bool, required=True,
                        help="True -- save imputed train and test sets, False -- do not save train and test sets")
    parser.add_argument("--evaluation_scenarios", type=str, help="a list of evaluation scenarios",
                        default=str(EVALUATION_SCENARIOS))

    args = parser.parse_args()
    args = validate_args(exp_config_obj=args, with_model_names=False)
    print('args.tune_imputers -- ', args.tune_imputers)

    print(
        f"Dataset name: {args.dataset}\n"
        f"Null imputers: {args.null_imputers}\n"
        f"Run nums: {args.run_nums}\n"
        f"Evaluation scenarios: {args.evaluation_scenarios}\n"
    )

    return args


if __name__ == '__main__':
    start_time = datetime.now()
    preconfigure_experiment()
    args = parse_input_args()

    benchmark = Benchmark(dataset_name=args.dataset,
                          null_imputers=args.null_imputers,
                          model_names=[])
    benchmark.impute_nulls_with_multiple_technique(run_nums=args.run_nums,
                                                   evaluation_scenarios=args.evaluation_scenarios,
                                                   tune_imputers=args.tune_imputers,
                                                   save_imputed_datasets=args.save_imputed_datasets)

    end_time = datetime.now()
    print(f'The script is successfully executed. Run time: {end_time - start_time}')
    print('Session UUID for all results:', benchmark._session_uuid)
