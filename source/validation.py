# TODO: missingness in train/test
def validate_args(exp_config_obj):
    """
    Validate parameters types and values in config yaml file.

    Extra details:
    * exp_config_obj.model_setting is an optional argument that defines a type of models to use
      to compute fairness and stability metrics. Default: 'batch'.

    * exp_config_obj.computation_mode is an optional argument that defines a non-default mode for metrics computation.
      Currently, only 'error_analysis' mode is supported.

    Parameters
    ----------
    exp_config_obj
        Object with parameters defined in a yaml file

    """
    # ================== Required parameters ==================
    if not isinstance(exp_config_obj.dataset_name, str):
        raise ValueError('dataset_name must be string')

    if not isinstance(exp_config_obj.bootstrap_fraction, float) \
            or exp_config_obj.bootstrap_fraction < 0.0 \
            or exp_config_obj.bootstrap_fraction > 1.0:
        raise ValueError('bootstrap_fraction must be float in [0.0, 1.0] range')

    if not isinstance(exp_config_obj.n_estimators, int) or exp_config_obj.n_estimators <= 1:
        raise ValueError('n_estimators must be integer greater than 1')

    if not isinstance(exp_config_obj.sensitive_attributes_dct, dict):
        raise ValueError('sensitive_attributes_dct must be python dictionary')

    if isinstance(exp_config_obj.sensitive_attributes_dct, dict):
        intersectional_attrs = [attr for attr in exp_config_obj.sensitive_attributes_dct.keys()
                                if INTERSECTION_SIGN in attr]
        for intersectional_attr in intersectional_attrs:
            intersectional_attr = intersectional_attr.strip()
            attrs = intersectional_attr.split(INTERSECTION_SIGN)
            attrs = [attr.strip() for attr in attrs]
            if len(attrs) != intersectional_attr.count(INTERSECTION_SIGN) + 1:
                raise ValueError(f"Incorrect format for an intersectional attribute name -- {intersectional_attr}."
                                 f"Intersectional signs must be between all attributes in this intersectional attribute.")

            for attr in attrs:
                if attr not in exp_config_obj.sensitive_attributes_dct.keys():
                    raise ValueError('Intersectional attributes in sensitive_attributes_dct must contain '
                                     'single sensitive attributes that also exist in sensitive_attributes_dct')

    # ================== Optional parameters ==================
    if exp_config_obj.model_setting is not None \
            and not isinstance(exp_config_obj.model_setting, str) \
            and exp_config_obj.model_setting not in ModelSetting:
        raise ValueError('model_setting must be a string that is included in the ModelSetting enum. '
                         'Refer to this function documentation for more details!')

    if exp_config_obj.computation_mode is not None \
            and not isinstance(exp_config_obj.computation_mode, str) \
            and exp_config_obj.computation_mode not in ComputationMode:
        raise ValueError('computation_mode must be a string that is included in the ComputationMode enum. '
                         'Refer to this function documentation for more details!')

    return True
