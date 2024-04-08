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

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
from scipy.stats import mode
from tensorflow.keras import Model
from autokeras import StructuredDataClassifier, StructuredDataRegressor

from source.utils.custom_logger import get_logger
from source.utils.dataframe_utils import get_mask


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

        self._statistics = dict()
        self._predictors: Dict[str, Model] = {}
        self.__logger = get_logger()

    def get_best_hyperparameters(self):

        super().get_best_hyperparameters()

        return {
            column: self._predictors[column].tuner.get_best_hyperparameters()[0].values
            for column in self._predictors.keys()
        }

    def fit(self, X: pd.DataFrame, target_columns: List[str]) -> BaseImputer:
        # Check if anything is actually missing and if not do not spend time on fitting
        missing_mask = X.isna()
        if not missing_mask.sum() > 0:
            self.__logger.warning("No missing value located; stop fitting.")
            return self

        super().fit(data=X, target_columns=target_columns)

        # Casting categorical columns to strings fixes problems
        # where categories are integer values and treated as regression task
        X = self._categorical_columns_to_string(X.copy())  # We don't want to change the input dataframe -> copy it

        # =============================================================================================================
        # 1) Make initial guess for missing values
        # =============================================================================================================
        self._statistics['col_medians'] = np.nanmedian(X[:, self._numerical_columns], axis=0) \
            if len(self._numerical_columns) >= 1 else None
        self._statistics['col_modes'] = mode(X[:, self._categorical_columns], axis=0, nan_policy='omit')[0] \
            if len(self._categorical_columns) >= 1 else None

        # =============================================================================================================
        # 2) Replace NaNs with median for numerica columns and modes for categorical columns
        # =============================================================================================================
        # Count missing per column
        col_missing_count = missing_mask.sum(axis=0)

        # Get col and row indices for missing
        missing_rows, missing_cols = np.where(missing_mask)

        # Replace NaNs in numerical columns
        if self._numerical_columns is not None:
            # Only keep indices for numerical vars
            keep_idx_num = np.in1d(missing_cols, self._numerical_columns)
            missing_num_rows = missing_rows[keep_idx_num]
            missing_num_cols = missing_cols[keep_idx_num]

            # Make initial guess for missing values
            col_medians = np.full(X.shape[1], fill_value=np.nan)
            col_medians[self._numerical_columns] = self._statistics['col_medians']
            X.loc[missing_num_rows, missing_num_cols] = np.take(col_medians, missing_num_cols)

        # Replace NaNs in categorical columns
        if self._categorical_columns is not None:
            # Only keep indices for categorical vars
            keep_idx_cat = np.in1d(missing_cols, self._categorical_columns)
            missing_cat_rows = missing_rows[keep_idx_cat]
            missing_cat_cols = missing_cols[keep_idx_cat]

            # Make initial guess for missing values
            col_modes = np.full(X.shape[1], fill_value=np.nan)
            col_modes[self._categorical_columns] = self._statistics['col_modes']
            X.loc[missing_cat_rows, missing_cat_cols] = np.take(col_modes, missing_cat_cols)

        # =============================================================================================================
        # 3) Create misscount_idx to sort indices of cols in X based on missing count
        # =============================================================================================================
        misscount_idx = np.argsort(col_missing_count)
        col_index = np.arange(X.shape[1])

        # =============================================================================================================
        # 4) Fit a predictor for each column with nulls. Start from the column with the smallest portion of nulls.
        # =============================================================================================================
        for s in misscount_idx:
            feature_cols = [c for c in self._categorical_columns + self._numerical_columns if c != s]

            # Column indices other than the one being imputed
            s_prime = np.delete(col_index, s)

            # Get indices of rows where 's' is observed and missing
            obs_rows = np.where(~missing_mask[:, s])[0]

            # Get observed values of 's'
            yobs = X[obs_rows, s]

            # Get observed 'X'
            xobs = X[np.ix_(obs_rows, s_prime)]

            # 5) Fit a random forest over observed and predict the missing
            if self._categorical_columns is not None and s in self._categorical_columns:
                StructuredDataModelSearch = StructuredDataClassifier
            else:
                StructuredDataModelSearch = StructuredDataRegressor

            self._predictors[s] = StructuredDataModelSearch(
                column_names=feature_cols,
                overwrite=True,
                max_trials=self.max_trials,
                tuner=self.tuner,
                directory="../models"
            )

            self._predictors[s].fit(
                x=xobs,
                y=yobs,
                epochs=self.epochs
            )
            # TODO: add predict

        self._fitted = True

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # Check if anything is actually missing and if not return original dataframe
        missing_mask = X.isna()
        if not missing_mask.sum() > 0:
            self.__logger.warning("No missing value located; stop fitting.")
            return X

        super().transform(data=X)

        # Save the original dtypes
        dtypes = X.dtypes

        # Casting categorical columns to strings fixes problems
        # where categories are integer values and treated as regression task
        X = self._categorical_columns_to_string(X.copy())  # We don't want to change the input dataframe -> copy it

        # Count missing per column
        col_missing_count = missing_mask.sum(axis=0)

        # Create misscount_idx to sort indices of cols in X based on missing count
        misscount_idx = np.argsort(col_missing_count)
        col_index = np.arange(X.shape[1])

        for s in misscount_idx:
            predictor = self._predictors[s]

            # Column indices other than the one being imputed
            s_prime = np.delete(col_index, s)

            # Get indices of rows where 's' is observed and missing
            mis_rows = np.where(missing_mask[:, s])[0]

            # If no missing, then skip
            if len(mis_rows) == 0:
                continue

            # Get missing 'X'
            xmis = X[np.ix_(mis_rows, s_prime)]

            # Predict ymis(s) using xmis(x)
            ymis = predictor.predict(xmis)[:, 0]
            # Update imputed matrix using predicted matrix ymis(s)
            X[mis_rows, s] = ymis

            self.__logger.info(f'Imputed nulls in column {s}')

        self._restore_dtype(X, dtypes)

        return X
