"""
The below code is based on this work:
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
import random
import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow.keras import Model
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
from autokeras import StructuredDataClassifier, StructuredDataRegressor

from source.utils.custom_logger import get_logger


def set_seed(seed: int) -> None:
    if seed:
        tf.random.set_seed(seed)
        random.seed(seed)
        np.random.seed(seed)


class ImputerError(Exception):
    """Exception raised for errors in Imputers"""
    pass


class BaseImputer(ABC):

    def __init__(self, seed: Optional[int] = None):
        """
        Abstract Base Class that defines the interface for all Imputer classes.

        Args:
            seed (Optional[int], optional): Seed to make behavior deterministic. Defaults to None.
        """

        self._fitted = False
        self._seed = seed

        set_seed(self._seed)

    @staticmethod
    def _guess_dtypes(data: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """
        Helper method: finds categorical and numerical columns.

        Args:
            data (pd.DataFrame): Data to guess the columns data types

        Returns:
            Tuple[List[str], List[str]]: Lists of categorical and numerical column names
        """

        categorical_columns = [c for c in data.columns if pd.api.types.is_categorical_dtype(data[c])]
        numerical_columns = [c for c in data.columns if pd.api.types.is_numeric_dtype(data[c]) and c not in categorical_columns]
        return categorical_columns, numerical_columns

    @staticmethod
    def _categorical_columns_to_string(data_frame: pd.DataFrame) -> pd.DataFrame:
        """
        Treats the categorical columns as strings and preserves missing values.

        Args:
            data_frame (pd.DataFrame): To-be-converted data

        Returns:
            pd.DataFrame: Data, where the categorical columns are strings
        """

        missing_mask = data_frame.isna()

        for column in data_frame.columns:
            if pd.api.types.is_categorical_dtype(data_frame[column]):
                data_frame[column] = data_frame[column].astype(str)

        # preserve missing values
        data_frame[missing_mask] = np.nan
        return data_frame

    def _restore_dtype(self, data: pd.DataFrame, dtypes: pd.Series) -> None:
        """
        Restores the data types of the columns

        Args:
            data (pd.DataFrame): Data, which column data types need to be restored
            dtypes (pd.Series): Data types
        """

        for column in data.columns:
            data[column] = data[column].astype(dtypes[column].name)

    @abstractmethod
    def get_best_hyperparameters(self) -> dict:
        """
        Returns the hyperparameters found as best during fitting.

        Returns:
            dict: Best hyperparameters
        """

        if not self._fitted:
            raise ImputerError("Imputer is not fitted.")

        return {}

    @abstractmethod
    def fit(self, data: pd.DataFrame, target_columns: List[str]):
        """
        Fit the imputer based on given `data` to imputed the `target_columns` later on.

        Args:
            data (pd.DataFrame): Data to train the imputer on.
            target_columns (List[str]): To-be-imputed columns.

        Raises:
            ImputerError: If `target_columns` is not a list.
            ImputerError: If element of `target_columns` is not column of `data`.
        """

        # some basic error checking
        if self._fitted:
            raise ImputerError(f"Imputer is already fitted. Target columns: {', '.join(self._target_columns)}")

        if not type(target_columns) == list:
            raise ImputerError(f"Parameter 'target_column' need to be of type list but is '{type(target_columns)}'")

        if any([column not in data.columns for column in target_columns]):
            raise ImputerError(f"All target columns ('{target_columns}') must be in: {', '.join(data.columns)}")

        self._target_columns = target_columns
        self._categorical_columns, self._numerical_columns = self._guess_dtypes(data)

    @abstractmethod
    def transform(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Imputes the columns of (the copied) `data` the imputer is fitted for.

        Args:
            data (pd.DataFrame): To-be-imputed data.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: First return value (index 0) is the (copied and) imputed `data`. \
                Second return value (index 1) is a mask representing which values are imputed. \
                It is a `DataFrame` because argument `target_columns` for `fit` method uses `list` of column names.
        """

        # some basic error checking
        if not self._fitted:
            raise ImputerError("Imputer is not fitted.")


class AutoMLImputer(BaseImputer):

    def __init__(
            self,
            max_trials: Optional[int] = 10,
            tuner: Optional[str] = None,
            validation_split: Optional[float] = 0.2,
            epochs: Optional[int] = 10,
            seed: Optional[int] = None
    ):
        """
        Deep Learning-learning based imputation mehtod. It uses AutoKeras to find good architecture/hyperparameters.

        Args:
            max_trials (Optional[int], optional): maximum number of trials for model selection. Defaults to 10.
            tuner (Optional[str], optional): AutoKeras hyperparameter tuning strategy. Defaults to None.
            validation_split (Optional[float], optional): validation split for AutoKeras fit. Defaults to 0.2.
            epochs (Optional[int], optional): number of epochs for AutoKeras fit. Defaults to 10.
            seed (Optional[int], optional): Seed to make behavior deterministic. Defaults to None.
        """

        super().__init__(
            seed=seed
        )

        self.max_trials = max_trials
        self.epochs = epochs
        self.validation_split = validation_split
        self.tuner = tuner
        self._predictors: Dict[str, Model] = {}
        self.__logger = get_logger()

    def get_best_hyperparameters(self):

        super().get_best_hyperparameters()

        return {
            column: self._predictors[column].tuner.get_best_hyperparameters()[0].values
            for column in self._predictors.keys()
        }

    def fit(self, data: pd.DataFrame, target_columns: List[str]) -> BaseImputer:

        super().fit(data=data, target_columns=target_columns)

        # cast categorical columns to strings fixes problems where categories are integer values and treated as regression task
        data = self._categorical_columns_to_string(data.copy())  # We don't want to change the input dataframe -> copy it

        for target_column in self._target_columns:

            missing_mask = data[target_column].isna()
            feature_cols = [c for c in self._categorical_columns + self._numerical_columns if c != target_column]

            if target_column in self._numerical_columns:
                StructuredDataModelSearch = StructuredDataRegressor

            elif target_column in self._categorical_columns:
                StructuredDataModelSearch = StructuredDataClassifier

            self._predictors[target_column] = StructuredDataModelSearch(
                column_names=feature_cols,
                overwrite=True,
                max_trials=self.max_trials,
                tuner=self.tuner,
                directory="../models"
            )

            self._predictors[target_column].fit(
                x=data.loc[~missing_mask, feature_cols],
                y=data.loc[~missing_mask, target_column],
                epochs=self.epochs
            )

        self._fitted = True

        return self

    def transform(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:

        super().transform(data=data)

        imputed_mask = data[self._target_columns].isna()

        # save the original dtypes because ..
        dtypes = data.dtypes

        # ... dtypes of data need to be same as for fitting
        data = self._categorical_columns_to_string(data.copy())  # We don't want to change the input dataframe -> copy it

        for target_column in self._target_columns:
            feature_cols = [c for c in self._categorical_columns + self._numerical_columns if c != target_column]
            missing_mask = data[target_column].isna()
            amount_missing_in_columns = missing_mask.sum()

            if amount_missing_in_columns > 0:
                data.loc[missing_mask, target_column] = self._predictors[target_column].predict(data.loc[missing_mask, feature_cols])[:, 0]
                self.__logger.debug(f'Imputed {amount_missing_in_columns} values in column {target_column}')

        self._restore_dtype(data, dtypes)

        return data, imputed_mask
