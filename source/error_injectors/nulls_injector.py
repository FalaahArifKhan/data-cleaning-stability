import numpy as np
import pandas as pd

from source.error_injectors.abstract_error_injector import AbstractErrorInjector


class NullsInjector(AbstractErrorInjector):
    def __init__(self, seed: int, strategy: str, columns_nulls_percentage_dct: dict):
        super().__init__(seed)
        self.strategy = strategy
        self.columns_nulls_percentage_dct = columns_nulls_percentage_dct
        
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
        for col, col_nulls_pct in self.columns_nulls_percentage_dct.items():
            if col not in df.columns:
                raise ValueError(f"Value caused the issue is {col}. "
                                 f"Keys in columns_nulls_percentage_dct must be the dataframe column names")
            if col_nulls_pct < 0 or col_nulls_pct > 1:
                raise ValueError(f"Value caused the issue is {col_nulls_pct}. "
                                 f"Column nulls percentage must be in [0.0-1.0] range.")

    def _validate_mar_input(self, df: pd.DataFrame):
        for col, dependent_cols_dict in self.columns_nulls_percentage_dct.items():
            if not isinstance(dependent_cols_dict, dict):
                raise ValueError(f"Invalid input for column '{col}'. It should be a dictionary.")
            for dep_col, dep_col_values_dict in dependent_cols_dict.items():
                if not isinstance(dep_col_values_dict, dict):
                    raise ValueError(f"Invalid input for dependent column '{dep_col}'. It should be a dictionary.")
                if dep_col not in df.columns:
                    raise ValueError(f"Value caused the issue is {dep_col}. "
                                     f"Keys in dependent_cols_dict must be the dataframe column names.")
                for dep_col_value, nulls_pct in dep_col_values_dict.items():
                    if nulls_pct < 0 or nulls_pct > 1:
                        raise ValueError(f"Invalid nulls percentage '{nulls_pct}' for value '{dep_col_value}' in column '{dep_col}'. Percentage should be a number between 0 and 1.")

    def _validate_mnar_input(self, df: pd.DataFrame):
        for col, col_values_dict in self.columns_nulls_percentage_dct.items():
            if not isinstance(col_values_dict, dict):
                raise ValueError(f"Invalid input for column '{col}'. It should be a dictionary.")
            if col not in df.columns:
                raise ValueError(f"Value caused the issue is {col}. "
                                 f"Keys in columns_nulls_percentage_dct must be the dataframe column names.")    
            for col_value, nulls_pct in col_values_dict.items():
                if nulls_pct < 0 or nulls_pct > 1:
                    raise ValueError(f"Invalid nulls percentage '{nulls_pct}' for value '{col_value}' in column '{col}'. Percentage should be a number between 0 and 1.")      
    
    def fit(self, df, target_column: str = None):
        self._validate_input_dicts(df)
        
    def _impute_nulls(self, df: pd.DataFrame, col_name: str, nulls_pct: float):
        if nulls_pct == 0:
            return df

        existing_nulls_count = df[col_name].isna().sum()
        target_nulls_count = int(df.shape[0] * nulls_pct)
        if existing_nulls_count > target_nulls_count:
            raise ValueError(f"Existing nulls count in '{col_name}' is greater than target nulls count. "
                             f"Increase nulls percentage for '{col_name}' to be greater than existing nulls percentage.")

        nulls_sample_size = target_nulls_count - existing_nulls_count
        notna_idxs = df[df[col_name].notna()].index
        np.random.seed(self.seed)
        random_row_idxs = np.random.choice(notna_idxs, size=nulls_sample_size, replace=False)
        df.loc[random_row_idxs, col_name] = None

        return df

    def _transform_mcar(self, df: pd.DataFrame):
        df_copy = df.copy(deep=True)
        for col_name, nulls_pct in self.columns_nulls_percentage_dct.items():
            df_copy = self._impute_nulls(df_copy, col_name, nulls_pct)
        return df_copy
    
    def _transform_mar(self, df: pd.DataFrame):
        df_copy = df.copy(deep=True)
        for col_name, dependent_cols_dict in self.columns_nulls_percentage_dct.items():
            for dependent_col_name, dependent_col_values_dict in dependent_cols_dict.items():
                for dependent_col_value, nulls_pct in dependent_col_values_dict.items():
                    df_subset = df_copy[df_copy[dependent_col_name] == dependent_col_value]
                    df_copy.loc[df_subset.index, col_name] = self._impute_nulls(df_subset, col_name, nulls_pct)[col_name]
        return df_copy
    
    def _transform_mnar(self, df: pd.DataFrame):
        df_copy = df.copy(deep=True)
        for col_name, col_values_dict in self.columns_nulls_percentage_dct.items():
            for col_value, nulls_pct in col_values_dict.items():
                df_subset = df_copy[df_copy[col_name] == col_value]
                df_copy.loc[df_subset.index, col_name] = self._impute_nulls(df_subset, col_name, nulls_pct)[col_name]
        return df_copy
                
    def transform(self, df: pd.DataFrame):
        self._validate_input_dicts(df)
        if self.strategy == 'MCAR':
            return self._transform_mcar(df)
        elif self.strategy == 'MAR':
            return self._transform_mar(df)
        elif self.strategy == 'MNAR':
            return self._transform_mnar(df)
        else:
            raise ValueError(f"Strategy '{self.strategy}' is not supported. Supported strategies are: MCAR, MAR, NMAR")

    def fit_transform(self, df, target_column: str = None):
        self.fit(df, target_column)
        transformed_df = self.transform(df)
        return transformed_df
