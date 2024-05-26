import os

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

from source.utils.pipeline_utils import encode_dataset_for_missforest

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
    tuning = kwargs['tune']
    computed_repaired_datasets_paths = kwargs['computed_repaired_datasets_paths']
    
    #val_set_ratio = 0.2
    val_set_ratio = 1000
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val,
                                                      test_size=val_set_ratio,
                                                      random_state=experiment_seed)
    
    X_train_with_nulls, _ = train_test_split(X_train_val_with_nulls,
                                            test_size=val_set_ratio,
                                            random_state=experiment_seed)
    
    assert X_train_with_nulls.index.isin(X_train.index).all(), \
        "Not all indexes of X_train_with_nulls are present in X_train"
        
    if tuning:
        train_set_cols_with_nulls = X_train_with_nulls.columns[X_train_with_nulls.isna().any()].tolist()
        train_numerical_null_columns = list(set(train_set_cols_with_nulls).intersection(numerical_columns))
        train_categorical_null_columns = list(set(train_set_cols_with_nulls).intersection(categorical_columns))
        
        X_train_list = read_repaired_datasets(X_train_with_nulls, computed_repaired_datasets_paths, 
                                              train_categorical_null_columns)
        
        model_params_list = tune_random_forest_for_boostclean(
                            X_train_list.values(), y_train, 
                            experiment_seed
                            )
    else:
        model_params_list = None
        
    boostclean_wrapper = BoostCleanWrapper(X_val=X_val,
                                           y_val=y_val,
                                           random_state=experiment_seed,
                                           tune=tuning,
                                           model_params_list=model_params_list,
                                           save_dir=save_dir,
                                           computed_repaired_datasets_paths=computed_repaired_datasets_paths)
    
    models_config = {
        ErrorRepairMethod.boost_clean.value: boostclean_wrapper
    }
    
    return models_config, X_train_with_nulls, y_train

def read_repaired_datasets(X_train, paths, categorical_columns_with_nulls):
        categorical_columns = X_train.select_dtypes(include=['object']).columns
        # Read imputed train sets from paths
        X_train_repairs = {}
        for path in paths:
            repair_method = os.path.basename(path)
            dataset_name = repair_method.split('_')[1]
            X_train_repaired = pd.read_csv(path, index_col=0)
            X_train_repaired_subset = X_train_repaired.loc[X_train.index]

            X_train_encoded = encode_dataset_for_boostclean(X_train_repaired_subset, categorical_columns)            
            X_train_repairs[repair_method] = X_train_encoded
            
        return X_train_repairs

def encode_dataset_for_boostclean(df, categorical_columns):
    """Encode categorical columns with OneHotEncoder."""
    df_enc = df.copy(deep=True)
    
    encoder = OneHotEncoder(drop='first')  # drop='first' to avoid multicollinearity
    column_transformer = ColumnTransformer(
    transformers=[
            ('onehot', encoder, categorical_columns)
        ],
        remainder='passthrough'  # This keeps the remaining columns as they are
    )
    
    X_encoded = column_transformer.fit_transform(df_enc) 
    return X_encoded 

def tune_random_forest_for_boostclean(X_train_list, y_train, random_state):
    print("Tuning Random Forest for BoostClean")
    params_grid = get_boostclean_params_for_tuning(random_state)
    classifier_params_grid = params_grid['RandomForestClassifier']['params'] 
    grid_search = GridSearchCV(
                estimator=params_grid['RandomForestClassifier']['model'],
                param_grid=classifier_params_grid,
                scoring="f1_macro",
                n_jobs=-1,
                cv=3
    )
    
    best_params_list = []
    for X_train in X_train_list:
        grid_search.fit(X=X_train, y=y_train)
        best_params_list.append({'fn': RandomForestClassifier, 'params': grid_search.best_params_})
        
    return best_params_list
    
def get_boostclean_params_for_tuning(models_tuning_seed):
    return {
        'RandomForestClassifier': {
            'model': RandomForestClassifier(random_state=models_tuning_seed),
            'params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 25, 50, 75, 100, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'bootstrap': [True, False]
            }
        }
        # 'RandomForestClassifier': {
        #     'model': RandomForestClassifier(random_state=models_tuning_seed),
        #     'params': {
        #         'n_estimators': [50],
        #         'max_depth': [10, 25],
        #         'min_samples_split': [2],
        #         'min_samples_leaf': [1],
        #         'bootstrap': [True]
        #     }
        # }
    }