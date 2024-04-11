"""
Evaluate models on the datasets without errors
"""
import os
import sys
import ast
from pathlib import Path

# Define a correct root path
sys.path.append(str(Path(f"{__file__}").parent.parent))
print('Current location: ', os.getcwd())

# Import dependencies
import argparse
import warnings
from datetime import datetime
from dotenv import load_dotenv

from source.custom_classes.benchmark import Benchmark


def preconfigure_experiment(env_file_path='../configs/secrets.env'):
    warnings.filterwarnings('ignore')
    os.environ["PYTHONWARNINGS"] = "ignore"

    # Load env variables
    load_dotenv(env_file_path)
    print('\n\nDB_NAME secret:', os.getenv("DB_NAME"))


def parse_input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="a dataset name", required=True)
    parser.add_argument("--models", type=str, help="a list of model names", required=True)
    parser.add_argument("--run_nums", type=str, help="a list of experiment run numbers", required=True)

    args = parser.parse_args()
    args.run_nums = ast.literal_eval(args.run_nums)
    args.models = ast.literal_eval(args.models)

    print(
        f"Dataset name: {args.dataset}\n"
        f"Models: {args.models}\n"
        f"Run nums: {args.run_nums}\n"
    )

    return args


if __name__ == '__main__':
    start_time = datetime.now()
    preconfigure_experiment()
    args = parse_input_args()

    benchmark = Benchmark(dataset_name=args.dataset,
                          null_imputers=[],
                          model_names=args.models)
    benchmark.evaluate_baselines(run_nums=args.run_nums,
                                 model_names=args.models)

    end_time = datetime.now()
    print(f'The script is successfully executed. Run time: {end_time - start_time}')
    print('Session UUID for all results:', benchmark._session_uuid)
