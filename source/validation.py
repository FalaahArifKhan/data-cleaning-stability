import ast

from configs.constants import ErrorRepairMethod, MLModels, EVALUATION_SCENARIOS
from configs.datasets_config import DATASET_CONFIG


def check_str_list_type(str_param):
    if '[' in str_param and ']' in str_param:
        return True
    return False


def is_in_enum(val, enum_obj):
    enum_vals = [member.value for member in enum_obj]
    return val in enum_vals


def validate_args(exp_config_obj, with_model_names=True):
    """
    Validate parameter types and values in the exp_config_obj.
    """
    # Check list types
    if not check_str_list_type(exp_config_obj.null_imputers):
        raise ValueError('null_imputers argument must be a list')

    if not check_str_list_type(exp_config_obj.run_nums):
        raise ValueError('run_nums argument must be a list')

    if not check_str_list_type(exp_config_obj.evaluation_scenarios):
        raise ValueError('evaluation_scenarios argument must be a list')

    # Cast to lists
    exp_config_obj.null_imputers = ast.literal_eval(exp_config_obj.null_imputers)
    exp_config_obj.run_nums = ast.literal_eval(exp_config_obj.run_nums)
    exp_config_obj.evaluation_scenarios = ast.literal_eval(exp_config_obj.evaluation_scenarios)

    # Check argument values
    if not DATASET_CONFIG.get(exp_config_obj.dataset, False):
        raise ValueError('dataset argument should be from the DATASET_CONFIG dictionary in configs/datasets_config.py')

    for null_imputer_name in exp_config_obj.null_imputers:
        if not is_in_enum(val=null_imputer_name, enum_obj=ErrorRepairMethod):
            raise ValueError('null_imputers argument should include values from the ErrorRepairMethod enum in configs/constants.py')

    for evaluation_scenario in exp_config_obj.evaluation_scenarios:
        if evaluation_scenario not in EVALUATION_SCENARIOS:
            raise ValueError('evaluation_scenarios argument should include values from the EVALUATION_SCENARIOS list in configs/constants.py')

    if with_model_names:
        if not check_str_list_type(exp_config_obj.models):
            raise ValueError('models argument must be a list')

        exp_config_obj.models = ast.literal_eval(exp_config_obj.models)

        for model_name in exp_config_obj.models:
            if not is_in_enum(val=model_name, enum_obj=MLModels):
                raise ValueError('models argument should include values from the MLModels enum in configs/constants.py')

    return exp_config_obj
