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
from tensorflow.keras import Model
from autokeras import StructuredDataClassifier, StructuredDataRegressor
from sklearn.metrics import mean_squared_error, precision_recall_fscore_support

from source.utils.custom_logger import get_logger
from source.utils.dataframe_utils import get_columns_sorted_by_nulls


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

        numerical_columns = [c for c in data.columns if pd.api.types.is_numeric_dtype(data[c])]
        categorical_columns = [c for c in data.columns if c not in numerical_columns]
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

        self._statistics = {'medians': dict(), 'modes': dict()}
        self._predictors: Dict[str, Model] = {}
        self.__logger = get_logger()

    def get_best_hyperparameters(self):
        super().get_best_hyperparameters()

        return {
            column: self._predictors[column].tuner.get_best_hyperparameters()[0].values
            for column in self._predictors.keys()
        }

    def fit(self, X: pd.DataFrame, target_columns: List[str], X_gt: pd.DataFrame = None, verbose: int = 1) -> BaseImputer:
        # Check if anything is actually missing and if not do not spend time on fitting
        missing_mask = X.isna()
        if not missing_mask.values.any():
            self.__logger.warning("No missing value located; stop fitting.")
            return self

        super().fit(data=X, target_columns=target_columns)
        print('Numerical columns:', self._numerical_columns)
        print('Categorical columns:', self._categorical_columns)

        X = X.copy(deep=True)

        # =============================================================================================================
        # 1) Make initial guess for missing values
        # =============================================================================================================
        # Replace NaNs in numerical columns
        if self._numerical_columns is not None:
            for column in self._numerical_columns:
                median = X[column].median(skipna=True)
                X[column].fillna(median, inplace=True)
                self._statistics['medians'][column] = median

        # Replace NaNs in categorical columns
        if self._categorical_columns is not None:
            for column in self._categorical_columns:
                mode_value = X[column].mode(dropna=True)[0]  # mode() returns a Series, [0] gets the mode value
                X[column].fillna(mode_value, inplace=True)
                self._statistics['modes'][column] = mode_value

        # =============================================================================================================
        # 2) Create a list of column names sorted by the number of nulls in them
        # =============================================================================================================
        sorted_columns_names_by_nulls = get_columns_sorted_by_nulls(missing_mask[self._target_columns])

        # =============================================================================================================
        # 3) Fit a predictor for each column with nulls. Start from the column with the smallest portion of nulls.
        # =============================================================================================================
        for target_column in sorted_columns_names_by_nulls:
            col_missing_mask = missing_mask[target_column]
            feature_cols = [c for c in self._categorical_columns + self._numerical_columns if c != target_column]

            if target_column in self._numerical_columns:
                StructuredDataModelSearch = StructuredDataRegressor

            elif target_column in self._categorical_columns:
                StructuredDataModelSearch = StructuredDataClassifier

            self._predictors[target_column] = StructuredDataModelSearch(
                column_names=feature_cols,
                multi_label=True if target_column in self._categorical_columns else None,
                overwrite=True,
                max_trials=self.max_trials,
                tuner=self.tuner,
                directory="../models"
            )
            self._predictors[target_column].fit(
                x=X.loc[~col_missing_mask, feature_cols],
                y=X.loc[~col_missing_mask, target_column],
                epochs=self.epochs,
                verbose=verbose
            )
            print('X.head(20):\n', X.head(20))
            print('X_gt.head(20):\n', X_gt.head(20))

            print('predictions:\n', self._predictors[target_column].predict(X.loc[col_missing_mask, feature_cols]))

            # Reuse predictions to improve performance of training for the later columns with nulls
            X.loc[col_missing_mask, target_column] = self._predictors[target_column].predict(X.loc[col_missing_mask, feature_cols])[:, 0]

            pred = X.loc[col_missing_mask, target_column]
            true = X_gt.loc[col_missing_mask, target_column]
            if target_column in self._categorical_columns:
                print('pred.head(20):\n', pred.head(20))
                print('true.head(20):\n', true.head(20))

                precision, recall, f1, _ = precision_recall_fscore_support(true, pred, average="micro")
                print('Precision for {}: {:.2f}'.format(target_column, precision))
                print('Recall for {}: {:.2f}'.format(target_column, recall))
                print('F1 score for {}: {:.2f}'.format(target_column, f1))
                print()

            else:
                rmse = mean_squared_error(true, pred, squared=False)
                print('RMSE for {}: {:.2f}'.format(target_column, rmse))

            self.__logger.info(f'Fitting for {target_column} column was successfully completed')

        self._fitted = True

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # Check if anything is actually missing and if not return original dataframe
        missing_mask = X.isna()
        print('missing_mask.sum():\n', missing_mask.sum())
        if not missing_mask.values.any():
            self.__logger.warning("No missing value located; stop fitting.")
            return X

        super().transform(data=X)
        X = X.copy(deep=True)

        # Make initial guess for missing values using collected statistics during self.fit()
        if self._numerical_columns is not None:
            for column in self._numerical_columns:
                median = self._statistics['medians'][column]
                X[column].fillna(median, inplace=True)

        if self._categorical_columns is not None:
            for column in self._categorical_columns:
                mode_value = self._statistics['modes'][column]
                X[column].fillna(mode_value, inplace=True)

        # Create a list of column names sorted by the number of nulls in them
        sorted_columns_names_by_nulls = get_columns_sorted_by_nulls(missing_mask[self._target_columns])
        print('2 X.isna().sum():\n', X.isna().sum())

        for target_column in sorted_columns_names_by_nulls:
            feature_cols = [c for c in self._categorical_columns + self._numerical_columns if c != target_column]
            col_missing_mask = missing_mask[target_column]
            amount_missing_in_columns = col_missing_mask.sum()

            if amount_missing_in_columns > 0:
                X.loc[col_missing_mask, target_column] = self._predictors[target_column].predict(X.loc[col_missing_mask, feature_cols])[:, 0]
                print(f'{target_column} column, X.loc[col_missing_mask, target_column]:\n{X.loc[col_missing_mask, target_column].head(20)}')
                self.__logger.info(f'Imputed {amount_missing_in_columns} values in column {target_column}')

        return X
