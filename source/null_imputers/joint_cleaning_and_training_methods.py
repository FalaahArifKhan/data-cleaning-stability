import pandas as pd
from sklearn.model_selection import train_test_split

from configs.constants import ErrorRepairMethod
from source.null_imputers.cpclean_wrapper import CPCleanWrapper
from source.null_imputers.boostclean_wrapper import BoostCleanWrapper


def prepare_cpclean(X_train_val: pd.DataFrame, y_train_val: pd.DataFrame, X_train_val_with_nulls: pd.DataFrame,
                    numerical_columns: list, categorical_columns: list, experiment_seed: int, **kwargs):
    save_dir = kwargs['save_dir']

    # Create a validation split
    optimal_validation_set_size = 1000
    val_set_ratio = 0.2
    val_size = min([optimal_validation_set_size, X_train_val.shape[0] * val_set_ratio])
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val,
                                                      test_size=val_size,
                                                      random_state=experiment_seed)
    X_train_with_nulls, _ = train_test_split(X_train_val_with_nulls,
                                             test_size=val_size,
                                             random_state=experiment_seed)

    # Ensure correctness of indexes in X_train_with_nulls and X_train sets
    assert X_train_with_nulls.index.isin(X_train.index).all(), \
        "Not all indexes of X_train_with_nulls are present in X_train"

    # Create a CPClean wrapper for Virny to conduct in-depth performance profiling
    cp_clean_wrapper = CPCleanWrapper(X_train_full=X_train,
                                      X_val=X_val,
                                      y_val=y_val,
                                      random_state=experiment_seed,
                                      save_dir=save_dir)
    models_config = {
        ErrorRepairMethod.cp_clean.value: cp_clean_wrapper
    }

    # Return a models config for Virny and sub-sampled train sets
    return models_config, X_train_with_nulls, y_train


def prepare_boostclean(X_train_val: pd.DataFrame, y_train_val: pd.DataFrame, X_train_val_with_nulls: pd.DataFrame,
                       numerical_columns: list, categorical_columns: list, experiment_seed: int, **kwargs):
    save_dir = kwargs['save_dir']
    
    val_set_ratio = 0.2
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val,
                                                      test_size=val_set_ratio,
                                                      random_state=experiment_seed)
    
    X_train_with_nulls, _ = train_test_split(X_train_val_with_nulls,
                                            test_size=val_set_ratio,
                                            random_state=experiment_seed)
    
    assert X_train_with_nulls.index.isin(X_train.index).all(), \
        "Not all indexes of X_train_with_nulls are present in X_train"
        
    boostclean_wrapper = BoostCleanWrapper(X_train_full=X_train,
                                           X_val=X_val,
                                           y_val=y_val,
                                           random_state=experiment_seed,
                                           save_dir=save_dir)
    
    models_config = {
        ErrorRepairMethod.boost_clean.value: boostclean_wrapper
    }
    
    return models_config, X_train_with_nulls, y_train