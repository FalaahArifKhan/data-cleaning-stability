import numpy as np
import pandas as pd
from scipy.stats import mode
import sys
import os
from pathlib import Path
sys.path.append(str(Path(f"{__file__}").parent.parent))

from sklearn.utils.validation import check_is_fitted, check_array

from kmodes.kprototypes import KPrototypes

from .abstract_null_imputer import AbstractNullImputer
from ..utils.dataframe_utils import _get_mask


class KMeansImputer(AbstractNullImputer):
    def __init__(self, seed: int, n_clusters: int, missing_values=np.nan,
                 max_iter: int = 100, n_init: int = 10, init: str = 'Cao', n_jobs: int = -1):
        super().__init__(seed)
        self.n_clusters = n_clusters
        self.missing_values = missing_values
        
        self.kprototypes = KPrototypes(
            n_clusters=n_clusters,
            max_iter=max_iter,
            init=init, n_init=n_init,
            n_jobs=n_jobs, random_state=seed
        )
        
    def _validate_input(self, df=None):
        # Check data integrity and calling arguments
        force_all_finite = False if self.missing_values in ["NaN",
                                                            np.nan] else True
        
        X = check_array(df, accept_sparse=False, dtype=np.float64,
                        force_all_finite=force_all_finite, copy=True)

        # Check for +/- inf
        if np.any(np.isinf(X)):
            raise ValueError("+/- inf values are not supported.")

        # Check if any column has all missing
        mask = _get_mask(X, self.missing_values)
        if np.any(mask.sum(axis=0) >= (X.shape[0])):
            raise ValueError("One or more columns have all rows missing.")
        
        return X, mask
    
    def fit(self, X, cat_vars, y=None):
        X, mask = self._validate_input(X)

        # Check cat_vars type and convert if necessary
        if cat_vars is not None:
            if type(cat_vars) == int:
                cat_vars = [cat_vars]
            elif type(cat_vars) == list or type(cat_vars) == np.ndarray:
                if np.array(cat_vars).dtype != int:
                    raise ValueError(
                        "cat_vars needs to be either an int or an array "
                        "of ints.")
            else:
                raise ValueError("cat_vars needs to be either an int or an array "
                                 "of ints.")

        # Identify numerical variables
        num_vars = np.setdiff1d(np.arange(X.shape[1]), cat_vars)
        num_vars = num_vars if len(num_vars) > 0 else None
        
        # Get col and row indices for missing
        _, missing_cols = np.where(mask)
        missing_cols = list(set(missing_cols))
        
        observed_columns = np.setdiff1d(np.arange(X.shape[1]), missing_cols)
        
        observed_cat_vars = np.intersect1d(observed_columns, cat_vars).tolist()
        observed_num_vars = np.intersect1d(observed_columns, num_vars).tolist()
        
        X_observed = np.hstack([X[:, observed_cat_vars], X[:, observed_num_vars]])
        
        self.cat_vars_ = list(range(len(observed_cat_vars)))
        self.num_vars_ = list(range(len(observed_cat_vars), len(observed_columns)))
        
        self.missing_cat_columns_ = np.intersect1d(missing_cols, cat_vars).tolist()
        self.missing_num_columns_ = np.intersect1d(missing_cols, num_vars).tolist()
        
        self.observed_columns_ = observed_columns
        self.missing_columns_ = missing_cols
        
        self.kprototypes.fit(X_observed, categorical=self.cat_vars_)
        
        return self
        
    
    def transform(self, X, y=None):
        # Confirm whether fit() has been called
        check_is_fitted(self, ["cat_vars_", "num_vars_"])
        
        X, mask = self._validate_input(X)  
        X_observed = X[:, self.observed_columns_]
        
        clusters = self.kprototypes.predict(X_observed, categorical=self.cat_vars_)
        
        for cluster in set(clusters):
            cluster_indices = np.where(clusters == cluster)[0]
            for col in self.missing_columns_:
                if col in self.missing_cat_columns_:
                    # calucate mode discarding nan and assign to missing values
                    X[cluster_indices, col] = mode(X[cluster_indices, col], axis=0, nan_policy='omit')[0]
                else:
                    X[cluster_indices, col] = np.nanmean(X[cluster_indices, col])
        
        return X
    
    
    def fit_transform(self, df, target_column: str = None, **fit_params):
        return self.fit(df, **fit_params).transform(df)
        