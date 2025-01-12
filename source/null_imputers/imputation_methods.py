import os
import copy
import logging
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.impute import SimpleImputer
import FrEIA.framework as Ff
import FrEIA.modules as Fm
from external_dependencies.azua.experiment.steps.train_step import run_train_main as azua_run_train_main
from external_dependencies.azua.datasets.dataset import Dataset
from external_dependencies.azua.datasets.variables import Variables
from external_dependencies.azua.datasets.csv_dataset_loader import CSVDatasetLoader

from source.null_imputers.automl_imputer import AutoMLImputer
from source.null_imputers.gain_imputer import GAINImputer
from source.null_imputers.missforest_imputer import MissForestImputer
from source.null_imputers.kmeans_imputer import KMeansImputer
from source.null_imputers.nomi_imputer import NOMIImputer
from source.null_imputers.tdm_imputer import TDMImputer
from source.null_imputers.hivae_imputer import HIVAEImputer
from source.utils.pipeline_utils import (encode_dataset_for_missforest, decode_dataset_for_missforest,
                                         encode_dataset_for_gain, decode_dataset_for_gain, encode_dataset_for_nomi,
                                         onehot_encode_dataset, onehot_decode_dataset, decode_dataset_for_mnar_pvae,
                                         encode_dataset_for_mnar_pvae, generate_types_csv)
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

    X_train_encoded, cat_encoders, _ = encode_dataset_for_nomi(df=X_train_with_nulls, dataset_name=dataset_name)
    X_tests_encoded_lst = [
        encode_dataset_for_nomi(df=X_test_with_nulls,
                                cat_encoders=cat_encoders,
                                dataset_name=dataset_name)[0]
        for X_test_with_nulls in X_tests_with_nulls_lst
    ]

    # Convert data to PyTorch tensors
    X_train_imputed_tensor = torch.tensor(X_train_encoded.values, dtype=torch.float32)
    X_tests_imputed_tensors_lst = [torch.tensor(X_test.values, dtype=torch.float32) for X_test in X_tests_encoded_lst]

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
    X_train_imputed_tensor, X_tests_imputed_tensors_lst = imputer.fit_transform(X_train=X_train_imputed_tensor,
                                                                                X_tests=X_tests_imputed_tensors_lst,
                                                                                verbose=True,
                                                                                report_interval=kwargs["report_interval"])

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


def impute_with_nomi(X_train_with_nulls: pd.DataFrame, X_tests_with_nulls_lst: list,
                     numeric_columns_with_nulls: list, categorical_columns_with_nulls: list,
                     hyperparams: dict, **kwargs):
    dataset_name = kwargs['dataset_name']
    seed = kwargs['experiment_seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    num_indices_with_nulls = [X_train_with_nulls.columns.get_loc(col) for col in numeric_columns_with_nulls]
    cat_indices_with_nulls = [X_train_with_nulls.columns.get_loc(col) for col in categorical_columns_with_nulls]

    # Encode categorical columns
    X_train_encoded, cat_encoders, _ = encode_dataset_for_nomi(df=X_train_with_nulls, dataset_name=dataset_name)
    X_tests_encoded_lst = [
        encode_dataset_for_nomi(df=X_test_with_nulls,
                                cat_encoders=cat_encoders,
                                dataset_name=dataset_name)[0]
        for X_test_with_nulls in X_tests_with_nulls_lst
    ]

    # Apply an imputer
    imputer = NOMIImputer(k_neighbors=kwargs['k_neighbors'],
                          similarity_metric=kwargs['similarity_metric'],
                          max_iterations=kwargs['max_iterations'],
                          tau=kwargs['tau'],
                          beta=kwargs['beta'])
    X_train_imputed_np = imputer.fit_transform(X_train_encoded.to_numpy(), num_indices_with_nulls, cat_indices_with_nulls)
    X_tests_imputed_np_lst = list(map(lambda X_test_encoded:
            imputer.transform(X_test_encoded.to_numpy(), num_indices_with_nulls, cat_indices_with_nulls),
      X_tests_encoded_lst)
    )

    # Convert numpy arrays back to DataFrames
    X_train_imputed = pd.DataFrame(X_train_imputed_np, columns=X_train_with_nulls.columns, index=X_train_with_nulls.index)
    X_tests_imputed_lst = [
        pd.DataFrame(X_test, columns=X_test_with_nulls.columns, index=X_test_with_nulls.index)
        for X_test, X_test_with_nulls in zip(X_tests_imputed_np_lst, X_tests_with_nulls_lst)
    ]

    # Decode categories back
    X_train_imputed = decode_dataset_for_missforest(X_train_imputed, cat_encoders, dataset_name=dataset_name)
    X_tests_imputed_lst = [
        decode_dataset_for_missforest(X_test_imputed, cat_encoders, dataset_name=dataset_name)
        for X_test_imputed in X_tests_imputed_lst
    ]

    hyperparams = {
        "k_neighbors": imputer.k_neighbors,
        "similarity_metric": imputer.similarity_metric,
        "max_iterations": imputer.max_iterations,
        "tau": imputer.tau,
        "beta": imputer.beta,
    }
    null_imputer_params_dct = {col: hyperparams for col in X_train_with_nulls.columns}

    return X_train_imputed, X_tests_imputed_lst, null_imputer_params_dct


def impute_with_notmiwae(X_train_with_nulls: pd.DataFrame, X_tests_with_nulls_lst: list,
                         numeric_columns_with_nulls: list, categorical_columns_with_nulls: list,
                         hyperparams: dict, **kwargs):
    from .notmiwae_imputer import train, not_imputationRMSE, notMIWAE, reverse_normalization

    def prepare_data_for_notmiwae(data: np.array, orig_mean=None, orig_std=None):
        N, D = data.shape
        dl = D - 1

        # ---- standardize data
        orig_mean = np.nanmean(data, axis=0) if orig_mean is None else orig_mean
        orig_std = np.nanstd(data, axis=0) if orig_std is None else orig_std
        data = data - orig_mean
        data = data / orig_std

        Xtrain = data.copy()
        Xval_org = data.copy()

        Xnan = Xtrain.copy()
        Xz = Xnan.copy()
        Xz[np.isnan(Xnan)] = 0

        S = np.array(~np.isnan(Xnan), dtype=np.float32)

        Xval = Xval_org.copy()
        Xvalz = Xval.copy()
        Xvalz[np.isnan(Xval)] = 0

        return Xtrain, Xnan, Xz, Xval, Xvalz, dl, S, orig_mean, orig_std

    directory = str(kwargs['directory'])
    dataset_name = kwargs['dataset_name']
    cat_indices_with_nulls = [X_train_with_nulls.columns.get_loc(col) for col in categorical_columns_with_nulls]

    # Encode categorical columns
    X_train_encoded, cat_encoders, _ = encode_dataset_for_nomi(df=X_train_with_nulls, dataset_name=dataset_name)
    X_tests_encoded_lst = [
        encode_dataset_for_nomi(df=X_test_with_nulls,
                                cat_encoders=cat_encoders,
                                dataset_name=dataset_name)[0]
        for X_test_with_nulls in X_tests_with_nulls_lst
    ]

    # Prepare data for not-MIWAE
    Xtrain, Xnan, Xz, Xval, Xvalz, dl, S, orig_mean, orig_std = prepare_data_for_notmiwae(X_train_encoded.to_numpy())

    # Apply an imputer
    imputer = notMIWAE(X=Xnan,
                       Xval=Xval,
                       n_latent=dl,
                       n_samples=kwargs["n_samples"],
                       n_hidden=kwargs["n_hidden"],
                       missing_process=kwargs["mprocess"],
                       name=directory)
    # Fit
    train(imputer, batch_size=kwargs["batch_size"], max_iter=kwargs["max_iter"], name=directory)

    # Transform train
    X_train_imputed_np = not_imputationRMSE(model=imputer,
                                            Xorg=Xtrain,
                                            Xz=Xz,
                                            X=Xnan,
                                            S=S,
                                            L=kwargs["L"],
                                            cat_indices=cat_indices_with_nulls)[1]
    X_train_imputed_np = reverse_normalization(X_train_imputed_np, orig_mean, orig_std)

    # Transform test
    X_tests_imputed_np_lst = []
    for X_test_encoded in X_tests_encoded_lst:
        Xtrain2, Xnan2, Xz2, Xval2, Xvalz2, dl2, S2, _, _ = prepare_data_for_notmiwae(X_test_encoded.to_numpy(),
                                                                                      orig_mean=orig_mean,
                                                                                      orig_std=orig_std)
        X_test_imputed_np = not_imputationRMSE(model=imputer,
                                               Xorg=Xtrain2,
                                               Xz=Xz2,
                                               X=Xnan2,
                                               S=S2,
                                               L=kwargs["L"],
                                               cat_indices=cat_indices_with_nulls)[1]
        X_test_imputed_np = reverse_normalization(X_test_imputed_np, orig_mean, orig_std)
        X_tests_imputed_np_lst.append(X_test_imputed_np)

    # Convert numpy arrays back to DataFrames
    X_train_imputed = pd.DataFrame(X_train_imputed_np, columns=X_train_with_nulls.columns, index=X_train_with_nulls.index)
    X_tests_imputed_lst = [
        pd.DataFrame(X_test, columns=X_test_with_nulls.columns, index=X_test_with_nulls.index)
        for X_test, X_test_with_nulls in zip(X_tests_imputed_np_lst, X_tests_with_nulls_lst)
    ]

    # Decode categories back
    X_train_imputed = decode_dataset_for_missforest(X_train_imputed, cat_encoders, dataset_name=dataset_name)
    X_tests_imputed_lst = [
        decode_dataset_for_missforest(X_test_imputed, cat_encoders, dataset_name=dataset_name)
        for X_test_imputed in X_tests_imputed_lst
    ]

    hyperparams = {
        "n_latent": imputer.n_latent,
        "n_hidden": imputer.n_hidden,
        "n_samples": imputer.n_samples,
        "out_dist": imputer.out_dist,
        "out_activation": imputer.out_activation,
        "embedding_size": imputer.embedding_size,
        "code_size": imputer.code_size,
        "missing_process": imputer.missing_process,
    }
    null_imputer_params_dct = {col: hyperparams for col in X_train_with_nulls.columns}

    return X_train_imputed, X_tests_imputed_lst, null_imputer_params_dct


def impute_with_mnar_pvae(X_train_with_nulls: pd.DataFrame, X_tests_with_nulls_lst: list,
                          numeric_columns_with_nulls: list, categorical_columns_with_nulls: list,
                          hyperparams: dict, **kwargs):
    directory = str(kwargs['directory'])
    dataset_name = kwargs['dataset_name']
    seed = kwargs['experiment_seed']

    # Encode categorical columns
    X_train_encoded, cat_encoders, scaler = encode_dataset_for_mnar_pvae(df=X_train_with_nulls, dataset_name=dataset_name)
    X_tests_encoded_lst = [
        encode_dataset_for_mnar_pvae(df=X_test_with_nulls,
                                     cat_encoders=cat_encoders,
                                     dataset_name=dataset_name,
                                     scaler=scaler)[0]
        for X_test_with_nulls in X_tests_with_nulls_lst
    ]

    # Prepare arguments for MNAR-PVAE
    model_type = "mnar_pvae"
    logger = logging.getLogger(model_type)
    model_config, train_hypers, impute_config = (
        kwargs["model_hyperparams"], kwargs["training_hyperparams"], kwargs["impute_config"])
    model_config["random_seed"] = seed

    # Prepare data for imputation
    train_data, train_mask = CSVDatasetLoader._process_data(X_train_encoded.to_numpy())
    X_tests_imputed_lst = list(map(
        lambda X_test_with_nulls: CSVDatasetLoader._process_data(X_test_with_nulls.to_numpy()), X_tests_encoded_lst)
    )

    variable_info = []
    for idx, col in enumerate(X_train_with_nulls.columns):
        lower_val, upper_val = X_train_encoded[col].min(), X_train_encoded[col].max()
        var = {
            "id": idx,  # Feature index, 0 to pixel count - 1
            "query": True,  # All features are query features.
            "type": "continuous",
            "name": col,  # Short variable description
            "lower": lower_val,  # Min feature value
            "upper": upper_val,  # Max feature value
        }
        variable_info.append(var)

    variables = Variables.create_from_dict({"variables": variable_info, "metadata_variables": []})
    dataset = Dataset(
        train_data=train_data,
        train_mask=train_mask,
        val_data=None,
        val_mask=None,
        test_data=None,
        test_mask=None,
        variables=variables,
        data_split=None,
    )

    # Perform imputation
    imputer = azua_run_train_main(
        logger=logger,
        model_type=model_type,
        output_dir=directory,
        variables=dataset.variables,
        dataset=dataset,
        device="cpu",
        model_config=model_config,
        train_hypers=train_hypers,
    )
    X_train_imputed_np = imputer.impute(train_data, train_mask, impute_config, vamp_prior_data=None)
    X_tests_imputed_np_lst = list(map(
        lambda test_data_tpl: imputer.impute(test_data_tpl[0], test_data_tpl[1], impute_config, vamp_prior_data=None), X_tests_imputed_lst)
    )

    # Convert numpy arrays to pandas dataframes
    X_train_imputed = pd.DataFrame(X_train_imputed_np, columns=X_train_with_nulls.columns, index=X_train_with_nulls.index)
    X_tests_imputed_lst = [
        pd.DataFrame(X_test, columns=X_test_with_nulls.columns, index=X_test_with_nulls.index)
        for X_test, X_test_with_nulls in zip(X_tests_imputed_np_lst, X_tests_with_nulls_lst)
    ]

    # Decode categories back
    X_train_imputed = decode_dataset_for_mnar_pvae(X_train_imputed, cat_encoders, dataset_name, scaler)
    X_tests_imputed_lst = [
        decode_dataset_for_mnar_pvae(X_test_imputed, cat_encoders, dataset_name, scaler)
        for X_test_imputed in X_tests_imputed_lst
    ]

    # Replace null values in df1 with values from df2
    X_train_imputed_final = copy.deepcopy(X_train_with_nulls)
    X_train_imputed_final = X_train_imputed_final.combine_first(X_train_imputed)
    X_tests_imputed_final_lst = [
        copy.deepcopy(X_test_with_nulls).combine_first(X_test_imputed) for X_test_with_nulls, X_test_imputed in zip(X_tests_with_nulls_lst, X_tests_imputed_lst)
    ]

    null_imputer_params_dct = {col: model_config for col in X_train_with_nulls.columns}
    return X_train_imputed_final, X_tests_imputed_final_lst, null_imputer_params_dct

def impute_with_hivae(X_train_with_nulls: pd.DataFrame, X_tests_with_nulls_lst: list,
                     numeric_columns_with_nulls: list, categorical_columns_with_nulls: list,
                     hyperparams: dict, **kwargs):
    """
    Impute missing values in the dataset using the HI-VAE model.

    Args:
        X_train_with_nulls (pd.DataFrame): Training data with missing values.
        X_tests_with_nulls_lst (list): List of test datasets with missing values.
        numeric_columns_with_nulls (list): List of numeric columns with missing values.
        categorical_columns_with_nulls (list): List of categorical columns with missing values.
        hyperparams (dict): Hyperparameters for the HI-VAE model.
        **kwargs: Additional arguments, including 'directory' for saving models and 'experiment_seed'.

    Returns:
        tuple: Imputed training dataset, list of imputed test datasets, and null imputer parameters.
    """
    # Extract additional arguments
    types_file = kwargs.get('types_file', 'types.csv')
    # Generate types.csv if not provided
    if not os.path.exists(types_file):
        generate_types_csv(X_train_with_nulls, types_file)

    # Split numeric and categorical columns in the training data
    numerical_columns = [c for c in X_train_with_nulls.columns if pd.api.types.is_numeric_dtype(X_train_with_nulls[c])]
    categorical_columns = [c for c in X_train_with_nulls.columns if c not in numerical_columns]

    # Deep copy datasets for processing
    X_train_imputed = X_train_with_nulls.copy()
    X_tests_imputed_lst = [X_test.copy() for X_test in X_tests_with_nulls_lst]

    # Encode categorical data for HI-VAE
    X_train_imputed, X_tests_imputed_lst = encode_dataset_for_gain(
        X_train=X_train_imputed,
        X_tests_lst=X_tests_imputed_lst,
        categorical_columns=categorical_columns
    )

    # Convert datasets to numpy arrays and create missing masks
    X_train_array = X_train_imputed.to_numpy()
    X_tests_array_lst = [X_test.to_numpy() for X_test in X_tests_imputed_lst]

    # Generate missing masks
    mask_train = ~pd.isna(X_train_array)
    masks_tests_lst = [~pd.isna(X_test_array) for X_test_array in X_tests_array_lst]

    # Initialize the HIVAE imputer
    imputer = HIVAEImputer(
        dim_latent_z=kwargs["training_hyperparams"].get('dim_latent_z', 2),
        dim_latent_y=kwargs["training_hyperparams"].get('dim_latent_y', 3),
        dim_latent_s=kwargs["training_hyperparams"].get('dim_latent_s', 4),
        batch_size=kwargs["training_hyperparams"].get('batch_size', 128),
        epochs=kwargs["training_hyperparams"].get('epochs', 100),
        learning_rate=kwargs["training_hyperparams"].get('learning_rate', 1e-3)
    )

    # Build the HI-VAE model
    imputer.build_model(types_file)

    # Train the model
    imputer.fit(X_train=X_train_array, mask_train=mask_train)

    # Impute the training data
    X_train_imputed_array = imputer.transform(X_test=X_train_array, mask_test=mask_train)

    # Impute the test datasets
    X_tests_imputed_array_lst = [
        imputer.transform(X_test=X_test_array, mask_test=mask_test)
        for X_test_array, mask_test in zip(X_tests_array_lst, masks_tests_lst)
    ]

    # Convert numpy arrays back to dataframes
    X_train_imputed = pd.DataFrame(X_train_imputed_array, columns=X_train_with_nulls.columns)
    X_tests_imputed_lst = [
        pd.DataFrame(X_test_imputed_array, columns=X_test.columns)
        for X_test_imputed_array, X_test in zip(X_tests_imputed_array_lst, X_tests_with_nulls_lst)
    ]

    # Decode categorical columns back to original format
    X_train_imputed, X_tests_imputed_lst = decode_dataset_for_gain(
        X_train=X_train_imputed,
        X_tests_lst=X_tests_imputed_lst,
        categorical_columns=categorical_columns
    )

    # Prepare null imputer parameters
    null_imputer_params_dct = {col: hyperparams for col in X_train_with_nulls.columns}

    return X_train_imputed, X_tests_imputed_lst, null_imputer_params_dct
