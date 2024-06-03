import numpy as np
import pandas as pd
from scipy.stats import mode
import sys
import os
from pathlib import Path
sys.path.append(str(Path(f"{__file__}").parent.parent))

from sklearn.utils.validation import check_is_fitted, check_array
from sklearn.model_selection import GridSearchCV

from kmodes.kprototypes import KPrototypes
from kmodes.kmodes import KModes

from .abstract_null_imputer import AbstractNullImputer
from ..utils.dataframe_utils import _get_mask


def get_kmeans_imputer_params_for_tuning(seed: int):
    return {
        "KMeansImputer": {
            "kprototypes_model": KPrototypes(random_state=seed),
            "kmodes_model": KModes(random_state=seed),
            "params": {
                "n_clusters": [2, 3, 4, 5, 6, 7, 8, 9, 10],
                "max_iter": [100, 200],
                "init": ["Huang", "Cao", "random"],
                "n_init": [1, 5, 10],
            }
            # "params": {
            #     "n_clusters": [2, 10],
            #     "max_iter": [100],
            #     "init": ["Cao"],
            #     "n_init": [1],
            # }
        }
    }


def custom_scoring_function(estimator, X, y=None):
    return estimator.epoch_costs_[-1]


class KMeansImputer(AbstractNullImputer):
    def __init__(self, imputer_mode: str,
                 seed: int, missing_values=np.nan,
                 n_jobs: int = -1, verbose: int = 0,
                 hyperparameters: dict = None):
        super().__init__(seed)
        self.missing_values = missing_values
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.hyperparameters = hyperparameters
        self.imputer_mode = imputer_mode
        
        if self.hyperparameters is not None:
            print("Hyperparameters are provided. Grid search will not be performed.")
            if imputer_mode == "kprototypes":
                self.model = KPrototypes(random_state=self.seed, n_jobs=self.n_jobs, **self.hyperparameters)
            elif imputer_mode == "kmodes":
                self.model = KModes(random_state=self.seed, n_jobs=self.n_jobs, **self.hyperparameters)
            else:
                raise ValueError("Invalid imputer mode. Choose either 'kprototypes' or 'kmodes'.")
        else:
            print("Hyperparameters are not provided. Grid search will be performed.")
            tuning_params = get_kmeans_imputer_params_for_tuning(seed)["KMeansImputer"]
            self.tuning_params = tuning_params["params"]
            if imputer_mode == "kprototypes":
                estimator = tuning_params["kprototypes_model"]
            elif imputer_mode == "kmodes":
                estimator = tuning_params["kmodes_model"]
            else:
                raise ValueError("Invalid imputer mode. Choose either 'kprototypes' or 'kmodes'.")
            
            self.model_grid_search = GridSearchCV(
                estimator=estimator,
                param_grid=self.tuning_params,
                scoring=custom_scoring_function,
                cv=3,
                n_jobs=self.n_jobs,
                verbose=self.verbose
            )
        
    def _validate_input(self, df=None):
        # Check data integrity and calling arguments
        force_all_finite = False if self.missing_values in ["NaN", np.nan] else True
        
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
        
        if self.hyperparameters is None:
            self.model_grid_search.fit(X_observed, categorical=self.cat_vars_)
            self.model = self.model_grid_search.best_estimator_
            self.best_params_ = self.model_grid_search.best_params_
        else:
            self.model.fit(X_observed, categorical=self.cat_vars_)
        
        pred_clusters = self.model.predict(X_observed, categorical=self.cat_vars_)
        self._calculate_cluster_stats(X, pred_clusters)
        # save percentage of clusters
        self.cluster_percentages_ = {str(cluster): len(np.where(pred_clusters == cluster)[0]) / len(pred_clusters) for cluster in set(pred_clusters)}
        print(f"Cluster percentages: {self.cluster_percentages_}")
        
        return self
    
    def _calculate_cluster_stats(self, X, clusters):
        self.cluster_statistics_ = {}
        for cluster in set(clusters):
            cluster_indices = np.where(clusters == cluster)[0]

            for col in self.missing_columns_:
                if col in self.missing_cat_columns_:
                    self.cluster_statistics_[(cluster, col)] = mode(X[cluster_indices, col], axis=0, nan_policy='omit')[0]
                else:
                    self.cluster_statistics_[(cluster, col)] = np.nanmean(X[cluster_indices, col])
        
        return self
    
    def transform(self, X, y=None):
        # Confirm whether fit() has been called
        check_is_fitted(self, ["cat_vars_", "num_vars_"])
        
        X, mask = self._validate_input(X)
        missing_rows, _ = np.where(mask)  
        X_observed = X[:, self.observed_columns_]
        
        clusters = self.model.predict(X_observed, categorical=self.cat_vars_)
        
        for cluster in set(clusters):
            cluster_indices = np.where(clusters == cluster)[0]
            missing_in_cluster_indices = np.intersect1d(cluster_indices, missing_rows)
            for col in self.missing_columns_:
                if col in self.missing_cat_columns_:
                    # calucate mode discarding nan and assign to missing values
                    X[missing_in_cluster_indices, col] = self.cluster_statistics_[(cluster, col)]
                else:
                    X[missing_in_cluster_indices, col] = self.cluster_statistics_[(cluster, col)]
        
        return X

    def fit_transform(self, df, target_column: str = None, **fit_params):
        return self.fit(df, **fit_params).transform(df)
    
    def get_predictors_params(self):
        if self.hyperparameters is None:
            output = self.best_params_
        else:
            output = self.hyperparameters
            
        return output
