import os
import numpy as np
import pandas as pd
from datetime import datetime
from virny.custom_classes.base_inprocessing_wrapper import BaseInprocessingWrapper

from external_dependencies.CPClean.training.knn import KNN
from external_dependencies.CPClean.cleaner.boost_clean import transform_y, train_classifiers
from external_dependencies.CPClean.repair.repair import repair
from external_dependencies.CPClean.training.preprocess import preprocess


class BoostCleanWrapper(BaseInprocessingWrapper):
    def __init__(self, X_train_full, X_val, y_val, random_state, save_dir, T=5, tune=False):
        self.X_train_full = X_train_full
        self.X_val = X_val
        self.y_val = y_val
        self.random_state = random_state
        self.save_dir = save_dir
        self.T = T
        self.tune = tune
        
        self.model_metadata = {
            "fn": KNN,
            "params": {"n_neighbors": 3}
        }
        
    def __copy__(self):
        return BoostCleanWrapper(X_train_full=self.X_train_full.copy(deep=False),
                                 X_val=self.X_val.copy(deep=False),
                                 y_val=self.y_val.copy(deep=False),
                                 random_state=self.random_state,
                                 save_dir=self.save_dir,
                                 T=self.T)
        
    def __deepcopy__(self, memo):
        return BoostCleanWrapper(X_train_full=self.X_train_full.copy(deep=True),
                                 X_val=self.X_val.copy(deep=True),
                                 y_val=self.y_val.copy(deep=True),
                                 random_state=self.random_state,
                                 save_dir=self.save_dir,
                                 T=self.T)
        
    def get_params(self):
        return {
            'random_state': self.random_state,
            'model_metadata_params': self.model_metadata['params'],
            'T': self.T
        }
        
    def set_params(self, random_state):
        return BoostCleanWrapper(X_train_full=self.X_train_full,
                                 X_val=self.X_val,
                                 y_val=self.y_val,
                                 random_state=random_state,
                                 save_dir=self.save_dir,
                                 T=self.T)
        
    def _fit_boost_clean(self, model, X_train_list, y_train, X_val, y_val, T=1):
        y_train = transform_y(y_train, 1)
        y_val = transform_y(y_val, 1)

        self.C_list = train_classifiers(X_train_list, y_train, model)
        N = len(y_val)
        W = np.ones((1, N)) / N

        preds_val = np.array([C.predict(X_val) for C in self.C_list]).T
        y_val = y_val.reshape(-1, 1)
        
        acc_list = (preds_val == y_val).astype(int)
        C_T = []
        a_T = []
        for t in range(self.T):
            acc_t = W.dot(acc_list)
            c_t = np.argmax(acc_t)

            e_c = 1 - acc_t[0, c_t]
            a_t = np.log((1-e_c)/(e_c+1e-8))
            
            C_T.append(c_t)
            a_T.append(a_t)
            
            for i in range(N):
                W[0, i] = W[0, i] * np.exp(-a_t * y_val[i, 0] * preds_val[i, c_t])

        self.a_T = np.array(a_T).reshape(1, -1)
        self.C_T = C_T

        preds_val = [C.predict(X_val) for C in self.C_list]
        preds_val_T = np.array([preds_val[c_t] for c_t in C_T])
        val_scores = self.a_T.dot(preds_val_T).T

        y_pred_val = np.sign(val_scores)
        val_acc = (y_pred_val == y_val).mean()

        return val_acc
    
    def _tune_boost_clean(self, model, X_train_list, y_train, X_val, y_val, T_range=[1, 5, 10]):
        y_train = transform_y(y_train, 1)
        y_val = transform_y(y_val, 1)

        self.C_list = train_classifiers(X_train_list, y_train, model)
        N = len(y_val)

        W_results = {}

        preds_val_list = [C.predict(X_val) for C in self.C_list]
        preds_val_array = np.array(preds_val_list).T
        y_val = y_val.reshape(-1, 1)
        acc_list = (preds_val_array == y_val).astype(int)
        
        for T in T_range:
            C_T = []
            a_T = []
            W = np.ones((1, N)) / N
            for t in range(T):
                acc_t = W.dot(acc_list)
                c_t = np.argmax(acc_t)

                e_c = 1 - acc_t[0, c_t]
                a_t = np.log((1-e_c)/(e_c+1e-8))
                
                C_T.append(c_t)
                a_T.append(a_t)
                
                for i in range(N):
                    second_term = y_val[i, 0]
                    third_term = preds_val_array[i, c_t]
                    W[0, i] = W[0, i] * np.exp(-a_t * second_term * third_term)

            a_T = np.array(a_T).reshape(1, -1)

            preds_val_T = np.array([preds_val_list[c_t] for c_t in C_T])
            val_scores = a_T.dot(preds_val_T).T

            y_pred_val = np.sign(val_scores)
            val_acc = (y_pred_val == y_val).mean()
            W_results[T] = [val_acc, a_T, C_T]
            print("###### Tuning BoostClean ######")
            print(f"BoostClean performance on a validation set with T={T}: {val_acc}")
            
        # Find the best T
        best_T = max(W_results, key=lambda x: W_results[x][0])
        self.T = best_T
        best_acc = W_results[best_T][0]
        self.a_T = W_results[best_T][1]
        self.C_T = W_results[best_T][2]
        
        return best_acc                
    
    def _predict_boost_clean(self, X_test):
        preds_test = [C.predict(X_test) for C in self.C_list]
        preds_test_T = np.array([preds_test[c_t] for c_t in self.C_T])
        test_scores = self.a_T.dot(preds_test_T).T

        y_pred_test = np.sign(test_scores)
        
        return y_pred_test
        
    def _build_dataset_objects(self, X_train_with_nulls, y_train):
        # X_train_clean will not be used but left for compatibility with preprocessing from CPClean
        X_train_clean = self.X_train_full.loc[X_train_with_nulls.index]
        ind_mv = X_train_with_nulls.isna()
        data_dct = {
            "X_train_clean": X_train_clean, "y_train": y_train,
            "X_train_dirty": X_train_with_nulls, "indicator": ind_mv,
            "X_full": None, "y_full": None,
            "X_val": self.X_val, "y_val": self.y_val,
        }

        return data_dct
        
    def fit(self, X_train, y_train):
        data_dct = self._build_dataset_objects(X_train, y_train)
        
        data_dct["X_train_repairs"] = repair(data_dct["X_train_dirty"], save_dir=self.save_dir)
        
        data_dct, self.preprocessor = preprocess(data_dct)
        
        if self.tune:
            boost_clean_result = self._tune_boost_clean(
                          model=self.model_metadata,
                          X_train_list=data_dct["X_train_repairs"].values(),
                          y_train=data_dct["y_train"],
                          X_val=data_dct["X_val"],
                          y_val=data_dct["y_val"]
                        )
        else:
            boost_clean_result = self._fit_boost_clean(
                                model=self.model_metadata,
                                X_train_list=data_dct["X_train_repairs"].values(),
                                y_train=data_dct["y_train"],
                                X_val=data_dct["X_val"],
                                y_val=data_dct["y_val"]
                                )
        
        print(f'BoostClean performance on a validation set: {boost_clean_result}')
        
        return self
    
    def predict(self, X):
        X_copy = X.copy(deep=True)
        X_preprocessed = self.preprocessor.transform(X_copy)
        
        return self._predict_boost_clean(X_preprocessed)
    
    def predict_proba(self, X):
        pass