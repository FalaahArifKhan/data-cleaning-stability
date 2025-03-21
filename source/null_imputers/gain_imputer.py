"""
The original paper for the code below:
- GitHub: https://github.com/se-jaeger/data-imputation-paper

- Citation:
@ARTICLE{imputation_benchmark_jaeger_2021,
	AUTHOR={Jäger, Sebastian and Allhorn, Arndt and Bießmann, Felix},
	TITLE={A Benchmark for Data Imputation Methods},
	JOURNAL={Frontiers in Big Data},
	VOLUME={4},
	PAGES={48},
	YEAR={2021},
	URL={https://www.frontiersin.org/article/10.3389/fdata.2021.693674},
	DOI={10.3389/fdata.2021.693674},
	ISSN={2624-909X},
	ABSTRACT={With the increasing importance and complexity of data pipelines, data quality became one of the key challenges in modern software applications. The importance of data quality has been recognized beyond the field of data engineering and database management systems (DBMSs). Also, for machine learning (ML) applications, high data quality standards are crucial to ensure robust predictive performance and responsible usage of automated decision making. One of the most frequent data quality problems is missing values. Incomplete datasets can break data pipelines and can have a devastating impact on downstream ML applications when not detected. While statisticians and, more recently, ML researchers have introduced a variety of approaches to impute missing values, comprehensive benchmarks comparing classical and modern imputation approaches under fair and realistic conditions are underrepresented. Here, we aim to fill this gap. We conduct a comprehensive suite of experiments on a large number of datasets with heterogeneous data and realistic missingness conditions, comparing both novel deep learning approaches and classical ML imputation methods when either only test or train and test data are affected by missing data. Each imputation method is evaluated regarding the imputation quality and the impact imputation has on a downstream ML task. Our results provide valuable insights into the performance of a variety of imputation methods under realistic conditions. We hope that our results help researchers and engineers to guide their data preprocessing method selection for automated data quality improvement.}
}
"""
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import optuna
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from .automl_imputer import BaseImputer

# Check if the required dependencies are available
if hasattr(tf, 'keras') and hasattr(tf.keras, 'Model'):
    from tensorflow import keras
    from tensorflow.compat.v1 import logging as tf_logging
    from tensorflow.keras import Model, Sequential
    from tensorflow.keras.activations import relu, sigmoid
    from tensorflow.keras.initializers import GlorotNormal
    from tensorflow.keras.layers import Dense, Input, Layer, concatenate
    from tensorflow.keras.optimizers import Adam

    tf.get_logger().setLevel('WARN')
    tf_logging.set_verbosity(tf_logging.ERROR)


logger = logging.getLogger()
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _merge_given_HPs_with_defaults(
        hyperparameter_grid: Dict[str, Dict[str, List[Union[int, float, bool]]]],
        default_hyperparameter_grid: Dict[str, Dict[str, List[Union[int, float, bool]]]]
) -> Dict[str, Dict[str, List[Union[int, float, bool]]]]:
    """
    If hyperparameter is given use it, else return default value. All others are ignored.
    """
    hyperparameters: Dict[str, Dict[str, List[Union[int, float, bool]]]] = dict()
    for hp_type in default_hyperparameter_grid.keys():
        hp_type = hp_type.lower()
        hyperparameters[hp_type] = {}
        for hp in default_hyperparameter_grid[hp_type].keys():
            hp = hp.lower()
            hyperparameters[hp_type][hp] = hyperparameter_grid.get(
                hp_type,
                default_hyperparameter_grid[hp_type]
            ).get(
                hp,
                default_hyperparameter_grid[hp_type][hp]
            )
    return hyperparameters


def _get_GAIN_search_space_for_grid_search(
        hyperparameter_grid: Dict[str, Dict[str, List[Union[int, float, bool]]]]
) -> Dict[str, List[Union[int, float, bool]]]:

    gain_default_hyperparameter_grid: Dict[str, Dict[str, List[Union[int, float, bool]]]] = {
        "gain": {
            "alpha": [100],
            "hint_rate": [0.9],
            "noise": [0.01]
        },
        "training": {
            "batch_size": [64],
            "max_epochs": [50],
            "early_stop": [3]
        },
        "generator": {
            "learning_rate": [0.0005],
            "beta_1": [0.9],
            "beta_2": [0.999],
            "epsilon": [1e-7],
            "amsgrad": [False]
        },
        "discriminator": {
            "learning_rate": [0.00005],
            "beta_1": [0.9],
            "beta_2": [0.999],
            "epsilon": [1e-7],
            "amsgrad": [False]
        }
    }

    hyperparameters = _merge_given_HPs_with_defaults(hyperparameter_grid, gain_default_hyperparameter_grid)

    search_space = dict(
        **hyperparameters["gain"],
        **hyperparameters["training"],
        **{f"generator_{key}": value for key, value in hyperparameters["generator"].items()},
        **{f"discriminator_{key}": value for key, value in hyperparameters["discriminator"].items()}
    )

    return search_space


class CategoricalEncoder(object):
    """
    Encoder only works on categorical columns. \
        It encodes the categorical values into `int` values fro `0` `n - 1`, where `n` is the number of categories.
    """

    def fit(self, data_frame: pd.DataFrame):
        """
        Creates attributes that map categorical values to integers and vice versa, separately for each column.

        Args:
            data_frame (pd.DataFrame): Data to fit mappings.

        Returns:
            CategoricalEncoder: Instance of itself.
        """

        self._numerical2category = dict()
        self._category2numerical = dict()

        for column in data_frame.columns:
            self._numerical2category[column] = {index: category for index, category in enumerate(data_frame[column].cat.categories)}
            self._category2numerical[column] = {category: index for index, category in enumerate(data_frame[column].cat.categories)}

        return self

    def transform(self, data_frame: pd.DataFrame) -> np.array:
        """
        Maps each column to their int representations.

        Args:
            data_frame (pd.DataFrame): To-be-encoded data

        Returns:
            np.array: Encoded data as matrix
        """

        data_frame = data_frame.copy()

        for column in data_frame.columns:
            data_frame.loc[:, column] = [self._category2numerical[column][value] if not pd.isna(value) else np.nan for value in data_frame[column]]

        return data_frame

    def inverse_transform(self, data_frame: pd.DataFrame) -> np.array:
        """
        Maps each int value to its categorical representations.

        Args:
            data_frame (pd.DataFrame): To-be-decoded data

        Returns:
            np.array: Decoded data as matrix
        """

        data_frame = data_frame.copy()

        for column in data_frame.columns:
            data_frame.loc[:, column] = [self._numerical2category[column][value] if not pd.isna(value) else np.nan for value in data_frame[column]]

        return data_frame

    def fit_transform(self, data_frame: pd.DataFrame) -> np.array:
        """
        Combines fitting of representations and returns (a copy) of their encoded values.

        Args:
            data_frame (pd.DataFrame): To-be-fitted and transformed data

        Returns:
            np.array: Encoded data as matrix
        """

        return self.fit(data_frame).transform(data_frame)


class GenerativeImputer(BaseImputer):

    def __init__(
            self,
            hyperparameter_grid: Dict[str, Dict[str, List[Union[int, float, bool]]]] = {},
            seed: Optional[int] = None
    ):

        super().__init__(seed=seed)

        self._hyperparameter_grid = hyperparameter_grid
        self._best_hyperparameters = None

    def _encode_data(self, data: pd.DataFrame) -> np.array:
        """
        Encode the input `DataFrame` into an `Array`. Categorical non numerical columns are first encoded, \
            then the whole matrix is scaled to be in range from `0` to `1`.

        Args:
            data (pd.DataFrame): To-be-imputed data

        Returns:
            np.array: To-be-imputed encoded data
        """

        if not self._fitted:
            if self._categorical_columns:

                self._data_encoder = CategoricalEncoder()
                data[self._categorical_columns] = self._data_encoder.fit_transform(data[self._categorical_columns])

            self._data_scaler = MinMaxScaler()
            data = self._data_scaler.fit_transform(data)

        else:
            if self._categorical_columns:

                data[self._categorical_columns] = self._data_encoder.transform(data[self._categorical_columns])

            data = self._data_scaler.transform(data)

        return data

    def _decode_encoded_data(self, encoded_data: np.array, columns: pd.Index, indices: pd.Index) -> pd.DataFrame:
        """
        Decodes the imputed data. Takes care of the proper column names and indices of the data.

        Args:
            encoded_data (np.array): Imputed data
            columns (pd.Index): column names of data
            indices (pd.Index): indices of data

        Returns:
            pd.DataFrame: Imputed data as `DataFrame`
        """

        data = self._data_scaler.inverse_transform(encoded_data)
        data = pd.DataFrame(data, columns=columns, index=indices)

        if self._categorical_columns:

            # round the encoded categories to next int. This is valid because we encode with OrdinalEncoder.
            # clip in range 0..(n-1), where n is the number of categories.
            for column in self._categorical_columns:
                data[column] = data[column].round(0)
                data[column] = data[column].clip(lower=0, upper=len(self._data_encoder._numerical2category[column].keys()) - 1)

            data[self._categorical_columns] = self._data_encoder.inverse_transform(data[self._categorical_columns])

        return data

    def _save_best_imputer(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
        if (trial.value and trial.number == study.best_trial.number) or self._best_hyperparameters is None:
            self.imputer.save(self._model_path, include_optimizer=False)
            self._best_hyperparameters = self.hyperparameters

    def get_best_hyperparameters(self) -> dict:

        super().get_best_hyperparameters()

        return self._best_hyperparameters


class EarlyStoppingExceeded(Exception):
    pass


class EarlyStopping:

    def __init__(self, early_stop: int):
        self.best_loss = None
        self.early_stop_count = 0
        self.early_stop = early_stop

    def update(self, new_loss: float):
        if self.best_loss is None:
            self.best_loss = new_loss
        elif self.best_loss > new_loss:
            self.best_loss = new_loss
            self.early_stop_count = 0
        else:
            self.early_stop_count += 1
            if self.early_stop_count >= self.early_stop:
                raise EarlyStoppingExceeded()


class GAINImputer(GenerativeImputer):

    def __init__(
            self,
            hyperparameter_grid: Dict[str, Dict[str, List[Union[int, float, bool]]]] = {},
            seed: Optional[int] = None,
            model_path: Optional[Path] = None
    ):
        """
        Implementation of Generative Adversarial Imputation Nets (GAIN): https://arxiv.org/abs/1806.02920

        Args:
            num_data_columns (int): Number of columns in the to-be-imputed data. Necessary to build the GAIN model properly.
            hyperparameter_grid (Dict[str, Union[int, float, Dict[str, Union[int, float]]]], optional): \
                Provides the hyperparameter grid used for HPO. The dictionary structure is as follows:
                hyperparameter_grid = {
                    "GAIN": {
                        "alpha": [...],
                        "hint_rate": [...],
                        "noise": [...]
                    },
                    "training": {
                        "batch_size": [...],
                        "max_epochs": [...],
                        "early_stop": [...],
                    },
                    "generator": {
                        "learning_rate": [...],
                        "beta_1": [...],
                        "beta_2": [...],
                        "epsilon": [...],
                        "amsgrad": [...]
                    },
                    "discriminator": {
                        "learning_rate": [...],
                        "beta_1": [...],
                        "beta_2": [...],
                        "epsilon": [...],
                        "amsgrad": [...]
                    }
                }
                Defaults to {}.
            seed (Optional[int], optional): To make process as deterministic as possible. Defaults to None.
            model_path (Optional[Path], optional): Path to save the model. Defaults to None.
        """

        super().__init__(
            hyperparameter_grid=hyperparameter_grid,
            seed=seed
        )

        self._model_path = model_path

    def _create_GAIN_model(self) -> None:
        """
        Helper method: creates the GAIN model based on the current hyperparameters.
        """

        # GAIN inputs
        X = Input((self._num_data_columns,))
        M = Input((self._num_data_columns,))
        H = Input((self._num_data_columns,))

        # GAIN structure
        self.generator = Sequential(
            [
                Dense(self._num_data_columns*2, activation=relu, kernel_initializer=GlorotNormal()),
                Dense(self._num_data_columns, activation=relu, kernel_initializer=GlorotNormal()),
                Dense(self._num_data_columns, activation=sigmoid, kernel_initializer=GlorotNormal())
            ]
        )

        self.discriminator = Sequential(
            [
                Dense(self._num_data_columns*2, activation=relu, kernel_initializer=GlorotNormal()),
                Dense(self._num_data_columns, activation=relu, kernel_initializer=GlorotNormal()),
                Dense(self._num_data_columns, activation=sigmoid, kernel_initializer=GlorotNormal())
            ]
        )

        gain_input = concatenate([X, M], axis=1)
        generater_output = self.generator(gain_input)
        intermediate = X * M + generater_output * (1 - M)
        intermediate_inputs = concatenate([intermediate, H], axis=1)
        discriminator_output = self.discriminator(intermediate_inputs)

        # GAIN loss
        discriminator_loss = -tf.reduce_mean(M * tf.math.log(discriminator_output + 1e-8) + (1 - M) * tf.math.log(1. - discriminator_output + 1e-8))
        generater_loss_temp = -tf.reduce_mean((1 - M) * tf.math.log(discriminator_output + 1e-8))

        MSE_loss = tf.reduce_mean((M * X - M * generater_output)**2) / tf.reduce_mean(M)
        generater_loss = generater_loss_temp + self.hyperparameters["alpha"] * MSE_loss

        self.gain = Model(inputs=[X, M, H], outputs=[generater_loss, discriminator_loss])

        # preserve original data and add generator output (i.e. imputed values)
        imputer_output = M * X + (1 - M) * generater_output
        self.imputer = Model(inputs=[X, M], outputs=[imputer_output])

    def _train_method(self, trial: optuna.trial.Trial, data: np.array) -> float:
        """
        Optuna's objective function. This is called multiple times by the Optuna framework. Goal is to minimize this method's return values

        Args:
            trial (optuna.trial.Trial): Optuna Trial, a process of evaluating an objective function
            data (np.array): Train data

        Returns:
            float: Generator loss
        """

        # Set hyperparameter once
        self._set_hyperparameters_for_optimization(trial)

        self._create_GAIN_model()

        generator_optimizer = Adam(**self.hyperparameters["generator_Adam"], clipvalue=1)
        discriminator_optimizer = Adam(**self.hyperparameters["discriminator_Adam"], clipvalue=1)

        generator_var_list = self.generator.trainable_weights
        discriminator_var_list = self.discriminator.trainable_weights

        @tf.function
        def train_step(X, M, H):
            with tf.GradientTape(persistent=True) as tape:
                generater_loss, discriminator_loss = self.gain([X, M, H])

            generator_gradients = tape.gradient(generater_loss, generator_var_list)
            generator_optimizer.apply_gradients(zip(generator_gradients, generator_var_list))

            discriminator_gradients = tape.gradient(discriminator_loss, discriminator_var_list)
            discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator_var_list))

            return generater_loss, discriminator_loss

        n_splits = 3
        k_fold = KFold(n_splits=n_splits, shuffle=True, random_state=self._seed)
        cv_generator_loss = 0

        for train_index, test_index in k_fold.split(data):

            train = tf.data.Dataset.from_tensor_slices(data[train_index])
            train_data = train.batch(self.hyperparameters["batch_size"]).prefetch(tf.data.AUTOTUNE)

            early_stopping = EarlyStopping(self.hyperparameters["early_stop"])

            for _ in range(self.hyperparameters["max_epochs"]):
                for train_batch in train_data:
                    X, M, H = self._prepare_GAIN_input_data(train_batch.numpy())
                    train_step(X, M, H)

                X_train, M_train, H_train = self._prepare_GAIN_input_data(data[train_index])
                epoch_loss, _ = self.gain([X_train, M_train, H_train])
                try:
                    early_stopping.update(epoch_loss)
                except EarlyStoppingExceeded:
                    break

            X, M, H = self._prepare_GAIN_input_data(data[test_index])
            generater_loss, _ = self.gain([X, M, H])
            cv_generator_loss += generater_loss

        # Optuna minimizes/maximizes this return value
        return cv_generator_loss / n_splits

    def _prepare_GAIN_input_data(self, data: np.array) -> Tuple[np.array, np.array, np.array]:
        """
        Prepare data to use it as GAIN input.

        Args:
            data (np.array): Encoded and (0, 1) scaled data

        Returns:
            Tuple[np.array, np.array, np.array]: Three matrices all of the same shapes used as GAIN input: \
                `X` (data matrix), `M` (mask matrix), `H` (hint matrix)
        """

        X_temp = np.nan_to_num(data, nan=0)
        Z_temp = np.random.uniform(0, self.hyperparameters["noise"], size=[data.shape[0], self._num_data_columns])

        random_hints = np.random.uniform(0., 1., size=[data.shape[0], self._num_data_columns])
        masked_random_hints = 1 * (random_hints < self.hyperparameters["hint_rate"])

        M = 1 - np.isnan(data)
        H = M * masked_random_hints
        X = M * X_temp + (1 - M) * Z_temp

        return X, M, H

    def _set_hyperparameters_for_optimization(self, trial: optuna.trial.Trial) -> None:
        """
        Helper method: samples hyperparameters for a HPO process

        Args:
            trial (optuna.trial.Trial): Optuna Trial, a process of evaluating an objective function
        """

        self.hyperparameters = {
            # GAIN
            "alpha": trial.suggest_discrete_uniform("alpha", 0, 9999999999, 1),
            "hint_rate": trial.suggest_discrete_uniform("hint_rate", 0, 1, 1),
            "noise": trial.suggest_discrete_uniform("noise", 0, 1, 1),

            # training
            # "batch_size": trial.suggest_discrete_uniform("batch_size", 0, 1024, 1),
            "batch_size": trial.suggest_discrete_uniform("batch_size", 0, 512, 1),
            "max_epochs": trial.suggest_discrete_uniform("max_epochs", 0, 10000, 1),
            "early_stop": trial.suggest_discrete_uniform("early_stop", 0, 1000, 1),

            # optimizers
            "generator_Adam": {
                "learning_rate": trial.suggest_discrete_uniform("generator_learning_rate", 0, 1, 1),
                "beta_1": trial.suggest_discrete_uniform("generator_beta_1", 0, 1, 1),
                "beta_2": trial.suggest_discrete_uniform("generator_beta_2", 0, 1, 1),
                "epsilon": trial.suggest_discrete_uniform("generator_epsilon", 0, 1, 1),
                "amsgrad": trial.suggest_categorical("generator_amsgrad", [True, False])
            },
            "discriminator_Adam": {
                "learning_rate": trial.suggest_discrete_uniform("discriminator_learning_rate", 0, 1, 1),
                "beta_1": trial.suggest_discrete_uniform("discriminator_beta_1", 0, 1, 1),
                "beta_2": trial.suggest_discrete_uniform("discriminator_beta_2", 0, 1, 1),
                "epsilon": trial.suggest_discrete_uniform("discriminator_epsilon", 0, 1, 1),
                "amsgrad": trial.suggest_categorical("discriminator_amsgrad", [True, False])
            }
        }

    def fit(self, data: pd.DataFrame, target_columns: List[str]) -> BaseImputer:

        super().fit(data=data, target_columns=target_columns)

        self._num_data_columns = data.shape[1]

        encoded_data = self._encode_data(data.copy())

        search_space = _get_GAIN_search_space_for_grid_search(self._hyperparameter_grid)
        study = optuna.create_study(sampler=optuna.samplers.GridSampler(search_space), direction="minimize")
        study.optimize(
            lambda trial: self._train_method(trial, encoded_data),
            callbacks=[self._save_best_imputer]
        )  # NOTE: n_jobs=-1 causes troubles because TensorFlow shares the graph across processes

        if self._model_path.exists():
            self.imputer = tf.keras.models.load_model(self._model_path, compile=False)
            shutil.rmtree(self._model_path, ignore_errors=True)

        self._fitted = True
        return self

    def transform(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:

        super().transform(data=data)

        imputed_mask = data[self._target_columns].isna()

        encoded_data = self._encode_data(data.copy())
        X, M, _ = self._prepare_GAIN_input_data(encoded_data)
        imputed = self.imputer([X, M]).numpy()

        # presever everything but the missing values in target columns.
        result = data.copy()
        imputed_data_frame = self._decode_encoded_data(imputed, data.columns, data.index)

        for column in imputed_mask.columns:
            result.loc[imputed_mask[column], column] = imputed_data_frame.loc[imputed_mask[column], column]

        return result, imputed_mask
