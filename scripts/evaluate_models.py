"""
Script for evaluating model performance on the imputed datasets from impute_nulls_with_predictor.py
"""
import os
import sys
from pathlib import Path

# Define a correct root path
sys.path.append(str(Path(f"{__file__}").parent.parent))
print('Current location: ', os.getcwd())

# Import dependencies
import ast
import argparse
import warnings
from datetime import datetime
from dotenv import load_dotenv

from configs.constants import EVALUATION_SCENARIOS
from source.custom_classes.benchmark import Benchmark


def preconfigure_experiment(env_file_path='./configs/secrets.env'):
    warnings.filterwarnings('ignore')
    os.environ["PYTHONWARNINGS"] = "ignore"

    # Load env variables
    load_dotenv(env_file_path)
    print('\n\nDB_NAME:', os.getenv("DB_NAME"))


def parse_input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="a dataset name", required=True)
    parser.add_argument("--null_imputers", type=str, help="a list of null imputers", required=True)
    parser.add_argument("--models", type=str, help="a list of model names", required=True)
    parser.add_argument("--run_nums", type=str, help="a list of experiment run numbers", required=True)
    parser.add_argument("--ml_impute", type=bool, required=True,
                        help="True -- apply ML-oriented imputers, False -- use pre-computed imputed datasets")
    parser.add_argument("--evaluation_scenarios", type=str, help="a list of evaluation scenarios",
                        default=str(EVALUATION_SCENARIOS))

    args = parser.parse_args()
    args.null_imputers = ast.literal_eval(args.null_imputers)
    args.models = ast.literal_eval(args.models)
    args.run_nums = ast.literal_eval(args.run_nums)
    args.evaluation_scenarios = ast.literal_eval(args.evaluation_scenarios)

    # TODO: args validation
    print(
        f"Dataset name: {args.dataset}\n"
        f"Null imputers: {args.null_imputers}\n"
        f"Models: {args.models}\n"
    )

    return args


if __name__ == '__main__':
    start_time = datetime.now()
    preconfigure_experiment()
    args = parse_input_args()

    benchmark = Benchmark(dataset_name=args.dataset,
                          null_imputers=args.null_imputers,
                          model_names=args.models)
    benchmark.run_experiment(run_nums=args.run_nums,
                             evaluation_scenarios=args.evaluation_scenarios,
                             model_names=args.models,
                             ml_impute=args.ml_impute)

    end_time = datetime.now()
    print(f'The script is successfully executed. Run time: {end_time - start_time}')
