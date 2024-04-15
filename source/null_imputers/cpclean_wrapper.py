import os
import numpy as np
from datetime import datetime
from virny.custom_classes.base_inprocessing_wrapper import BaseInprocessingWrapper

from external_dependencies.CPClean.utils import makedir
from external_dependencies.CPClean.repair.repair import repair
from external_dependencies.CPClean.training.preprocess import preprocess
from external_dependencies.CPClean.training.knn import KNN
from external_dependencies.CPClean.cleaner.CPClean.clean import CPClean
from external_dependencies.CPClean.cleaner.CPClean.debugger import Debugger

from source.utils.common_helpers import generate_base64_hash


class CPCleanWrapper(BaseInprocessingWrapper):
    def __init__(self, X_train_full, X_val, y_val, random_state, save_dir):
        self.X_train_full = X_train_full
        self.X_val = X_val
        self.y_val = y_val
        self.random_state = random_state
        self.save_dir = save_dir

        self.n_jobs = os.cpu_count()
        self.model_metadata = {
            "fn": KNN,
            "params": {"n_neighbors": 3}
        }
        self.cleaner = CPClean(K=self.model_metadata["params"]["n_neighbors"],
                               n_jobs=self.n_jobs,
                               random_state=self.random_state)

        self.preprocessor = None  # will be created during fitting

    def __copy__(self):
        return CPCleanWrapper(X_train_full=self.X_train_full.copy(deep=False),
                              X_val=self.X_val.copy(deep=False),
                              y_val=self.y_val.copy(deep=False),
                              random_state=self.random_state,
                              save_dir=self.save_dir)

    def __deepcopy__(self, memo):
        return CPCleanWrapper(X_train_full=self.X_train_full.copy(deep=True),
                              X_val=self.X_val.copy(deep=True),
                              y_val=self.y_val.copy(deep=True),
                              random_state=self.random_state,
                              save_dir=self.save_dir)

    def get_params(self):
        return {
            'random_state': self.random_state,
            'n_jobs': self.n_jobs,
            'model_metadata_params': self.model_metadata['params'],
        }

    def _build_dataset_objects(self, X_train_with_nulls, y_train):
        X_train_clean = self.X_train_full.loc[X_train_with_nulls.index]
        ind_mv = X_train_with_nulls.isna()
        print('X_train_with_nulls.isna().sum().sum():', X_train_with_nulls.isna().sum().sum())
        data_dct = {
            "X_train_clean": X_train_clean, "y_train": y_train,
            "X_train_dirty": X_train_with_nulls, "indicator": ind_mv,
            "X_full": None, "y_full": None,
            "X_val": self.X_val, "y_val": self.y_val,
        }

        return data_dct

    def _fit_cp_clean(self, data, method, sample_size):
        X_train_repairs = np.array([data["X_train_repairs"][m] for m in data["repair_methods"]])

        datetime_now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_hash = generate_base64_hash()
        debugger = Debugger(data, self.model_metadata, makedir([self.save_dir, 'logs', f'{datetime_now_str}_{random_hash}']))

        X_train_mean = data["X_train_repairs"]["mean_mode"] \
            if "mean_mode" in data["X_train_repairs"] else data["X_train_repairs"]["mean"]
        self.cleaner.fit(X_train_repairs, data["y_train"], data["X_val"], data["y_val"],
                         gt=data["X_train_gt"], X_train_mean=X_train_mean,
                         debugger=debugger, restore=False, method=method, sample_size=sample_size)

        val_acc, _, _, val_f1 = self.cleaner.score(data["X_val"], data["y_val"])
        cp_result = {"val_acc": val_acc, "val_f1": val_f1, "percent_clean": debugger.percent_clean}

        return cp_result

    def fit(self, X_train, y_train):
        # Build synthetic dataset objects required for CPClean
        data_dct = self._build_dataset_objects(X_train, y_train)

        # Created repaired datasets applying different imputation methods
        data_dct["X_train_repairs"] = repair(data_dct["X_train_dirty"], save_dir=None)

        # Preprocess train and test sets
        data_dct, self.preprocessor = preprocess(data_dct)

        # Fit CPClean
        cp_result = self._fit_cp_clean(data=data_dct,
                                       method="sgd_cpclean",
                                       sample_size=64)
        print(f'CPClean performance on a validation set: {cp_result}')

        return self

    def predict_proba(self, X):
        pass

    def predict(self, X):
        X_preprocessed, _ = self.preprocessor.transform(X)
        return self.cleaner.predict(X_preprocessed)
