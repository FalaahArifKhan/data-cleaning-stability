from typing import List, Dict, Tuple, Any

import numpy as np
import pandas as pd

from .abstract_error_injector import AbstractErrorInjector


class NullsInjector(AbstractErrorInjector):
    def __init__(self, seed: int, strategy: str, columns_with_nulls: list, null_percentage: float, condition: Tuple[str, Any] = None):
        super().__init__(seed)
        np.random.seed(seed)
        self.strategy = strategy
        self.columns_with_nulls = columns_with_nulls
        self.null_percentage = null_percentage
        self.condition = condition

    def _validate_input_dicts(self, df: pd.DataFrame):
        if self.strategy not in ['MCAR', 'MAR', 'MNAR']:
            raise ValueError(f"Invalid strategy '{self.strategy}'. Supported strategies are: MCAR, MAR, NMAR")

        if self.strategy == 'MCAR':
            self._validate_mcar_input(df)
        elif self.strategy == 'MAR':
            self._validate_mar_input(df)
        elif self.strategy == 'MNAR':
            self._validate_mnar_input(df)

    def _validate_mcar_input(self, df: pd.DataFrame):
        for col in self.columns_with_nulls:
            if col not in df.columns:
                raise ValueError(f"Value caused the issue is {col}. "
                                 f"Keys in columns_nulls_percentage_dct must be the dataframe column names")
            if self.null_percentage < 0 or self.null_percentage > 1:
                raise ValueError(f"Value caused the issue is {self.null_percentage}. "
                                 f"Column nulls percentage must be in [0.0-1.0] range.")

    def _validate_mar_input(self, df: pd.DataFrame):
        self._validate_mcar_input(df)
        if not isinstance(self.condition, tuple):
            raise ValueError(f"Invalid input for condition '{self.condition}'. It should be a tuple.")
        if len(self.condition) != 2:
            raise ValueError(f"Invalid input for condition '{self.condition}'. It should be a tuple with 2 elements.")
        if self.condition[0] not in df.columns:
            raise ValueError(f"Value caused the issue is {self.condition[0]}. "
                             f"Keys in condition must be the dataframe column names.")

    def _validate_mnar_input(self, df: pd.DataFrame):
        self._validate_mar_input(df)
        if len(self.columns_with_nulls) != 1:
            raise ValueError(f"Invalid input for columns_with_nulls '{self.columns_with_nulls}'. It should be a list with 1 element.")
        if self.columns_with_nulls[0] != self.condition[0]:
            raise ValueError(f"Invalid input for columns_with_nulls '{self.columns_with_nulls}'. It should be the same as the condition column.")

    def _get_condition_by_symbol(self, df, condition_col, symbol, condition_val):
        if symbol == 'ge':
            return df[condition_col] >= condition_val
        elif symbol == 'gt':
            return df[condition_col] > condition_val
        elif symbol == 'le':
            return df[condition_col] <= condition_val
        elif symbol == 'lt':
            return df[condition_col] < condition_val

    def _filter_df_by_condition(self, df: pd.DataFrame, condition_col: str, condition_val, include_val: bool):
        if isinstance(condition_val, list):
            df_condition = df[condition_col].isin(condition_val) if include_val else ~df[condition_col].isin(condition_val)

        elif isinstance(condition_val, dict):
            # Validate condition
            symbol_counts = {'g': 0, 'l': 0}
            for key in condition_val.keys():
                if key not in ('ge', 'gt', 'le', 'lt'):
                    raise ValueError(f"Condition symbol {key} is not in ('ge', 'gt', 'le', 'lt')")
                symbol_counts[key[0]] += 1

            if symbol_counts['g'] > 1 or symbol_counts['l'] > 1:
                raise ValueError(f"Condition should not include more than one greater symbol "
                                 f"or more than one less symbol")

            df_condition = None
            for symbol in condition_val.keys():
                val = condition_val[symbol]
                cur_df_condition = self._get_condition_by_symbol(df=df,
                                                                 condition_col=condition_col,
                                                                 symbol=symbol,
                                                                 condition_val=val)
                if df_condition is None:
                    df_condition = cur_df_condition
                else:
                    df_condition &= cur_df_condition

        else:
            df_condition = df[condition_col] == condition_val if include_val else df[condition_col] != condition_val

        return df[df_condition]

    def _inject_nulls(self, df: pd.DataFrame):
        df_copy = df.copy(deep=True)

        existing_nulls_count = df_copy[self.columns_with_nulls].isna().any().sum()
        target_nulls_count = int(df_copy.shape[0] * self.null_percentage)

        if existing_nulls_count > target_nulls_count:
            raise ValueError(f"Existing nulls count in '{self.columns_with_nulls}' is greater than target nulls count. "
                             f"Increase nulls percentage for '{self.columns_with_nulls}' to be greater than existing nulls percentage.")

        nulls_sample_size = target_nulls_count - existing_nulls_count
        notna_idxs = df_copy[df_copy[self.columns_with_nulls].notna()].index

        random_row_idxs = np.random.choice(notna_idxs, size=nulls_sample_size, replace=False)
        random_columns = np.random.choice(self.columns_with_nulls, size=nulls_sample_size, replace=True)

        random_sample_df = pd.DataFrame({'column': random_columns, 'random_idx': random_row_idxs})
        for idx, col_name in enumerate(self.columns_with_nulls):
            col_random_row_idxs = random_sample_df[random_sample_df['column'] == col_name]['random_idx'].values
            if col_random_row_idxs.shape[0] == 0:
                continue

            df_copy.loc[col_random_row_idxs, col_name] = np.nan

        return df_copy

    def _inject_nulls_mcar(self, df: pd.DataFrame):
        return self._inject_nulls(df)

    def _inject_nulls_mar(self, df: pd.DataFrame):
        df_copy = df.copy(deep=True)
        condition_col, condition_value = self.condition
        subset_df = self._filter_df_by_condition(df=df_copy,
                                                 condition_col=condition_col,
                                                 condition_val=condition_value,
                                                 include_val=True)
        subset_df_injected = self._inject_nulls(subset_df)
        df_copy.loc[subset_df_injected.index, :] = subset_df_injected

        return df_copy

    def _inject_nulls_mnar(self, df: pd.DataFrame):
        df_copy = df.copy(deep=True)
        condition_col, condition_value = self.condition
        subset_df = self._filter_df_by_condition(df=df_copy,
                                                 condition_col=condition_col,
                                                 condition_val=condition_value,
                                                 include_val=True)
        subset_df_injected = self._inject_nulls(subset_df)
        df_copy.loc[subset_df_injected.index, :] = subset_df_injected

        return df_copy

    def transform(self, df: pd.DataFrame):
        self._validate_input_dicts(df)
        if self.strategy == 'MCAR':
            return self._inject_nulls_mcar(df)
        elif self.strategy == 'MAR':
            return self._inject_nulls_mar(df)
        elif self.strategy == 'MNAR':
            return self._inject_nulls_mnar(df)
        else:
            raise ValueError(f"Strategy '{self.strategy}' is not supported. Supported strategies are: MCAR, MAR, NMAR")

    def fit(self, df, target_column: str = None):
        self._validate_input_dicts(df)

    def fit_transform(self, df, target_column: str = None):
        self.fit(df, target_column)
        transformed_df = self.transform(df)

        return transformed_df
