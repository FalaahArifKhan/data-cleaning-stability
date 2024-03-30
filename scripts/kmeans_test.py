import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans

from virny.datasets.data_loaders import ACSIncomeDataset

module_path = os.path.abspath(os.path.join('.'))
if module_path not in sys.path:
    sys.path.append(module_path)
    
from error_injectors.random_nulls_injector_strategies_v2 import RandomNullsInjectorStrategies
from cleaners.missforest import MissForest


class KMeansImputer:
    def __init__(self, n_clusters=2, max_iter=300, seed=42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.seed = seed
        self.kmeans = None
        self.cluster_labels = None
        self.cluster_centers = None
        
    def fit(self, X):
        self.kmeans = KMeans(n_clusters=self.n_clusters, max_iter=self.max_iter, random_state=self.seed)
        self.cluster_labels = self.kmeans.fit_predict(X)
        self.cluster_centers = self.kmeans.cluster_centers_
        
    def transform(self, X):
        X_copy = X.copy()
        for col in X_copy.columns:
            X_copy[col] = X_copy[col].fillna(X_copy.groupby(self.cluster_labels)[col].transform('mean'))
        return X_copy
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    
    
    
data_loader = ACSIncomeDataset(state=['GA'], year=2018, with_nulls=False,
                               subsample_size=20_000, subsample_seed=42)

mcar_injector = RandomNullsInjectorStrategies(seed=42, strategy='MCAR', columns_with_nulls=["AGEP", "SCHL", "MAR"], null_percentage=0.3)
injected_df = mcar_injector.transform(data_loader.X_data)

kmeans_imputer = KMeansImputer(n_clusters=2, max_iter=300, seed=42)
repaired_df = kmeans_imputer.fit_transform(injected_df)
print(repaired_df.isnull().mean() * 100)