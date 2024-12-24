import copy
import pandas as pd
import torch
import torch.nn as nn
from sklearn.impute import SimpleImputer
import FrEIA.framework as Ff
import FrEIA.modules as Fm

from source.null_imputers.automl_imputer import AutoMLImputer
from source.null_imputers.gain_imputer import GAINImputer
from source.null_imputers.missforest_imputer import MissForestImputer
from source.null_imputers.kmeans_imputer import KMeansImputer
from source.null_imputers.tdm_imputer import TDMImputer
from source.utils.pipeline_utils import (encode_dataset_for_missforest, decode_dataset_for_missforest,
                                         encode_dataset_for_gain, decode_dataset_for_gain)
from source.utils.dataframe_utils import get_numerical_columns_indexes


def impute_with_deletion(X_train_with_nulls: pd.DataFrame, X_tests_with_nulls_lst: list,
                         numeric_columns_with_nulls: list, categorical_columns_with_nulls: list,
                         hyperparams: dict, **kwargs):
    X_train_imputed = copy.deepcopy(X_train_with_nulls)
    X_tests_imputed_lst = list(map(lambda X_test_with_nulls: copy.deepcopy(X_test_with_nulls), X_tests_with_nulls_lst))

    # Apply deletion for a train set
    X_train_imputed = X_train_imputed.dropna()

    # Apply median-mode for a test set
    num_imputer = SimpleImputer(strategy='median')
    num_imputer.fit(X_train_imputed[numeric_columns_with_nulls])
    for i in range(len(X_tests_imputed_lst)):
        X_tests_imputed_lst[i][numeric_columns_with_nulls] = num_imputer.transform(X_tests_imputed_lst[i][numeric_columns_with_nulls])

    cat_imputer = SimpleImputer(strategy='most_frequent')
    cat_imputer.fit(X_train_imputed[categorical_columns_with_nulls])
    for i in range(len(X_tests_imputed_lst)):
        X_tests_imputed_lst[i][categorical_columns_with_nulls] = cat_imputer.transform(X_tests_imputed_lst[i][categorical_columns_with_nulls])

    null_imputer_params_dct = None
    return X_train_imputed, X_tests_imputed_lst, null_imputer_params_dct


def impute_with_simple_imputer(X_train_with_nulls: pd.DataFrame, X_tests_with_nulls_lst: list,
                               numeric_columns_with_nulls: list, categorical_columns_with_nulls: list,
                               hyperparams: dict, **kwargs):
    X_train_imputed = copy.deepcopy(X_train_with_nulls)
    X_tests_imputed_lst = list(map(lambda X_test_with_nulls: copy.deepcopy(X_test_with_nulls), X_tests_with_nulls_lst))

    # Impute numerical columns
    num_imputer = SimpleImputer(strategy=kwargs['num'])
    X_train_imputed[numeric_columns_with_nulls] = num_imputer.fit_transform(X_train_imputed[numeric_columns_with_nulls])
    for i in range(len(X_tests_imputed_lst)):
        X_tests_imputed_lst[i][numeric_columns_with_nulls] = num_imputer.transform(X_tests_imputed_lst[i][numeric_columns_with_nulls])

    # Impute categorical columns
    cat_imputer = SimpleImputer(strategy=kwargs['cat'], fill_value='missing') \
        if kwargs['cat'] == 'constant' else SimpleImputer(strategy=kwargs['cat'])
    X_train_imputed[categorical_columns_with_nulls] = cat_imputer.fit_transform(X_train_imputed[categorical_columns_with_nulls])
    for i in range(len(X_tests_imputed_lst)):
        X_tests_imputed_lst[i][categorical_columns_with_nulls] = cat_imputer.transform(X_tests_imputed_lst[i][categorical_columns_with_nulls])

    null_imputer_params_dct = None
    return X_train_imputed, X_tests_imputed_lst, null_imputer_params_dct


def impute_with_automl(X_train_with_nulls: pd.DataFrame, X_tests_with_nulls_lst: list,
                       numeric_columns_with_nulls: list, categorical_columns_with_nulls: list,
                       hyperparams: dict, **kwargs):
    directory = kwargs['directory']
    seed = kwargs['experiment_seed']
    target_columns = list(set(numeric_columns_with_nulls) | set(categorical_columns_with_nulls))

    X_train_imputed = copy.deepcopy(X_train_with_nulls)
    X_tests_imputed_lst = list(map(lambda X_test_with_nulls: copy.deepcopy(X_test_with_nulls), X_tests_with_nulls_lst))

    imputer = AutoMLImputer(max_trials=kwargs["max_trials"],
                            tuner=kwargs["tuner"],
                            validation_split=kwargs["validation_split"],
                            epochs=kwargs["epochs"],
                            seed=seed,
                            directory=directory)
    imputer.fit(X=X_train_imputed,
                target_columns=target_columns,
                verbose=0)

    X_train_imputed = imputer.transform(X_train_imputed)
    X_tests_imputed_lst = list(map(lambda X_test_imputed: imputer.transform(X_test_imputed), X_tests_imputed_lst))

    null_imputer_params_dct = imputer.get_best_hyperparameters()
    return X_train_imputed, X_tests_imputed_lst, null_imputer_params_dct


def impute_with_gain(X_train_with_nulls: pd.DataFrame, X_tests_with_nulls_lst: list,
                     numeric_columns_with_nulls: list, categorical_columns_with_nulls: list,
                     hyperparams: dict, **kwargs):
    directory = kwargs['directory']
    seed = kwargs['experiment_seed']
    target_columns = list(set(numeric_columns_with_nulls) | set(categorical_columns_with_nulls))
    numerical_columns = [c for c in X_train_with_nulls.columns if pd.api.types.is_numeric_dtype(X_train_with_nulls[c])]
    categorical_columns = [c for c in X_train_with_nulls.columns if c not in numerical_columns]

    X_train_imputed = copy.deepcopy(X_train_with_nulls)
    X_tests_imputed_lst = list(map(lambda X_test_with_nulls: copy.deepcopy(X_test_with_nulls), X_tests_with_nulls_lst))
    X_train_imputed, X_tests_imputed_lst = encode_dataset_for_gain(X_train=X_train_imputed,
                                                                   X_tests_lst=X_tests_imputed_lst,
                                                                   categorical_columns=categorical_columns)

    imputer = GAINImputer(hyperparameter_grid=kwargs["hyperparameter_grid"],
                          seed=seed,
                          model_path=directory)
    imputer.fit(data=X_train_imputed, target_columns=target_columns)

    X_train_imputed, _ = imputer.transform(X_train_imputed)
    X_tests_imputed_lst = list(map(lambda X_test_imputed: imputer.transform(X_test_imputed)[0], X_tests_imputed_lst))

    X_train_imputed, X_tests_imputed_lst = decode_dataset_for_gain(X_train=X_train_imputed,
                                                                   X_tests_lst=X_tests_imputed_lst,
                                                                   categorical_columns=categorical_columns)
    null_imputer_params_dct = {col: imputer.hyperparameters for col in X_train_with_nulls.columns}
    return X_train_imputed, X_tests_imputed_lst, null_imputer_params_dct


def impute_with_tdm(X_train_with_nulls: pd.DataFrame, X_tests_with_nulls_lst: list,
                    numeric_columns_with_nulls: list, categorical_columns_with_nulls: list,
                    hyperparams: dict, **kwargs):
    dataset_name = kwargs['dataset_name']
    seed = kwargs['experiment_seed']
    torch.manual_seed(seed)  # Set the random seed for reproducibility

    X_train_encoded, cat_encoders, _ = encode_dataset_for_missforest(df=X_train_with_nulls,
                                                                     dataset_name=dataset_name,
                                                                     categorical_columns_with_nulls=categorical_columns_with_nulls)
    X_tests_imputed_lst = [
        encode_dataset_for_missforest(df=X_test_with_nulls,
                                      cat_encoders=cat_encoders,
                                      dataset_name=dataset_name,
                                      categorical_columns_with_nulls=categorical_columns_with_nulls)[0]
        for X_test_with_nulls in X_tests_with_nulls_lst
    ]

    # Convert data to PyTorch tensors
    X_train_imputed_tensor = torch.tensor(X_train_encoded.values, dtype=torch.float32)
    X_tests_imputed_tensors_lst = [torch.tensor(X_test.values, dtype=torch.float32) for X_test in X_tests_imputed_lst]

    # Create a projector
    n, d = X_train_imputed_tensor.shape
    k = kwargs['network_width']
    def subnet_fc(dims_in, dims_out):
        return nn.Sequential(nn.Linear(dims_in, k * d), nn.SELU(),
                             nn.Linear(k * d, k * d), nn.SELU(),
                             nn.Linear(k * d,  dims_out))
    projector = Ff.SequenceINN(d)
    for _ in range(kwargs['network_depth']):
        projector.append(Fm.RNVPCouplingBlock, subnet_constructor=subnet_fc)

    imputer = TDMImputer(projector=projector,
                         im_lr=kwargs["lr"],
                         proj_lr=kwargs["lr"],
                         niter=kwargs["niter"],
                         batchsize=kwargs["batchsize"])
    imputer.fit(X_train=X_train_imputed_tensor, verbose=True, report_interval=kwargs["report_interval"])

    X_train_imputed_tensor = imputer.transform(X_train_imputed_tensor)
    X_tests_imputed_tensors_lst = list(map(lambda X_test_imputed: imputer.transform(X_test_imputed), X_tests_imputed_tensors_lst))

    # Convert tensors back to DataFrames
    X_train_imputed = pd.DataFrame(X_train_imputed_tensor.numpy(), columns=X_train_with_nulls.columns, index=X_train_with_nulls.index)
    X_tests_imputed_lst = [
        pd.DataFrame(X_test.numpy(), columns=X_test_with_nulls.columns, index=X_test_with_nulls.index)
        for X_test, X_test_with_nulls in zip(X_tests_imputed_tensors_lst, X_tests_with_nulls_lst)
    ]

    # Decode categories back
    X_train_imputed = decode_dataset_for_missforest(X_train_imputed, cat_encoders, dataset_name=dataset_name)
    X_tests_imputed_lst = [
        decode_dataset_for_missforest(X_test_imputed, cat_encoders, dataset_name=dataset_name)
        for X_test_imputed in X_tests_imputed_lst
    ]

    hyperparams = {
        "im_lr": imputer.im_lr,
        "proj_lr": imputer.proj_lr,
        "niter": imputer.niter,
        "batchsize": imputer.batchsize,
        "n_pairs": imputer.n_pairs,
        "noise": imputer.noise,
    }
    null_imputer_params_dct = {col: hyperparams for col in X_train_with_nulls.columns}

    return X_train_imputed, X_tests_imputed_lst, null_imputer_params_dct


def impute_with_missforest(X_train_with_nulls: pd.DataFrame, X_tests_with_nulls_lst: list,
                           numeric_columns_with_nulls: list, categorical_columns_with_nulls: list,
                           hyperparams: dict, **kwargs):
    seed = kwargs['experiment_seed']
    dataset_name = kwargs['dataset_name']

    # Impute numerical columns
    missforest_imputer = MissForestImputer(seed=seed, hyperparams=hyperparams)

    X_train_encoded, cat_encoders, categorical_columns_idxs = encode_dataset_for_missforest(X_train_with_nulls,
                                                                                            dataset_name=dataset_name,
                                                                                            categorical_columns_with_nulls=categorical_columns_with_nulls)
    X_train_repaired_values = missforest_imputer.fit_transform(X_train_encoded.values.astype(float), cat_vars=categorical_columns_idxs)
    X_train_repaired = pd.DataFrame(X_train_repaired_values, columns=X_train_encoded.columns, index=X_train_encoded.index)
    X_train_imputed = decode_dataset_for_missforest(X_train_repaired, cat_encoders, dataset_name=dataset_name)

    X_tests_imputed_lst = []
    for i in range(len(X_tests_with_nulls_lst)):
        X_test_with_nulls = X_tests_with_nulls_lst[i]

        X_test_encoded, _, _ = encode_dataset_for_missforest(X_test_with_nulls,
                                                             cat_encoders=cat_encoders,
                                                             dataset_name=dataset_name,
                                                             categorical_columns_with_nulls=categorical_columns_with_nulls)
        X_test_repaired_values = missforest_imputer.transform(X_test_encoded.values.astype(float))
        X_test_repaired = pd.DataFrame(X_test_repaired_values, columns=X_test_encoded.columns, index=X_test_encoded.index)
        X_test_imputed = decode_dataset_for_missforest(X_test_repaired, cat_encoders, dataset_name=dataset_name)

        X_tests_imputed_lst.append(X_test_imputed)

    if hyperparams is not None:
        null_imp_params_dct = {col: hyperparams for col in X_train_with_nulls.columns}
    else:
        predictor_params = missforest_imputer.get_predictors_params()
        null_imp_params_dct = {X_train_with_nulls.columns[i]: {str(k): predictor_params[i][k] for k in predictor_params[i]} for i in predictor_params}
    
    return X_train_imputed, X_tests_imputed_lst, null_imp_params_dct


def impute_with_kmeans(X_train_with_nulls: pd.DataFrame, X_tests_with_nulls_lst: list,
                       numeric_columns_with_nulls: list, categorical_columns_with_nulls: list,
                       hyperparams: dict, **kwargs):
    seed = kwargs['experiment_seed']
    dataset_name = kwargs['dataset_name']

    # Set an appropriate kmeans_imputer_mode type
    numerical_columns_idxs = get_numerical_columns_indexes(X_train_with_nulls)
    if len(numerical_columns_idxs) == len(numeric_columns_with_nulls):
        kmeans_imputer_mode = "kmodes"
    else:
        kmeans_imputer_mode = "kprototypes"

    X_train_encoded, cat_encoders, categorical_columns_idxs = \
        encode_dataset_for_missforest(X_train_with_nulls,
                                      dataset_name=dataset_name,
                                      categorical_columns_with_nulls=categorical_columns_with_nulls)

    # Impute numerical columns
    kmeans_imputer = KMeansImputer(seed=seed, imputer_mode=kmeans_imputer_mode, hyperparameters=hyperparams)
    
    X_train_repaired_values = kmeans_imputer.fit_transform(X_train_encoded.values.astype(float), cat_vars=categorical_columns_idxs)
    X_train_repaired = pd.DataFrame(X_train_repaired_values, columns=X_train_encoded.columns, index=X_train_encoded.index)
    X_train_imputed = decode_dataset_for_missforest(X_train_repaired, cat_encoders, dataset_name=dataset_name)

    X_tests_imputed_lst = []
    for i in range(len(X_tests_with_nulls_lst)):
        X_test_with_nulls = X_tests_with_nulls_lst[i]

        X_test_encoded, _, _ = encode_dataset_for_missforest(X_test_with_nulls,
                                                             cat_encoders=cat_encoders,
                                                             dataset_name=dataset_name,
                                                             categorical_columns_with_nulls=categorical_columns_with_nulls)
        X_test_repaired_values = kmeans_imputer.transform(X_test_encoded.values.astype(float))
        X_test_repaired = pd.DataFrame(X_test_repaired_values, columns=X_test_encoded.columns, index=X_test_encoded.index)
        X_test_imputed = decode_dataset_for_missforest(X_test_repaired, cat_encoders, dataset_name=dataset_name)

        X_tests_imputed_lst.append(X_test_imputed)
    
    null_imp_params_dct = {col: kmeans_imputer.get_predictors_params() for col in X_train_with_nulls.columns}
    return X_train_imputed, X_tests_imputed_lst, null_imp_params_dct
