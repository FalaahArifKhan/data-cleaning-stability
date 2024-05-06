import sys
from pathlib import Path

# Define a correct root path
sys.path.append(str(Path(f"{__file__}").parent.parent.parent))

import warnings
from collections import defaultdict

import numpy as np
from scipy.stats import mode

from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted, check_array
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV

from .abstract_null_imputer import AbstractNullImputer
from ..utils.dataframe_utils import _get_mask


def get_missforest_params_for_tuning(models_tuning_seed):
    return {
        'RandomForestClassifier': {
            'model': RandomForestClassifier(random_state=models_tuning_seed),
            'params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 25, 50, 75, 100, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'bootstrap': [True, False]
            }
        },
        'RandomForestRegressor': {
            'model': RandomForestRegressor(random_state=models_tuning_seed),
            'params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 25, 50, 75, 100, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'bootstrap': [True, False]
            }
        }
    }


class MissForestImputer(BaseEstimator, AbstractNullImputer):
    """Missing value imputation using Random Forests.

    MissForest imputes missing values using Random Forests in an iterative
    fashion. By default, the imputer begins imputing missing values of the
    column (which is expected to be a variable) with the smallest number of
    missing values -- let's call this the candidate column.
    The first step involves filling any missing values of the remaining,
    non-candidate, columns with an initial guess, which is the column mean for
    columns representing numerical variables and the column mode for columns
    representing categorical variables. After that, the imputer fits a random
    forest model with the candidate column as the outcome variable and the
    remaining columns as the predictors over all rows where the candidate
    column values are not missing.
    After the fit, the missing rows of the candidate column are
    imputed using the prediction from the fitted Random Forest. The
    rows of the non-candidate columns act as the input data for the fitted
    model.
    Following this, the imputer moves on to the next candidate column with the
    second smallest number of missing values from among the non-candidate
    columns in the first round. The process repeats itself for each column
    with a missing value, possibly over multiple iterations or epochs for
    each column, until the stopping criterion is met.
    The stopping criterion is governed by the "difference" between the imputed
    arrays over successive iterations. For numerical variables (num_vars_),
    the difference is defined as follows:

     sum((X_new[:, num_vars_] - X_old[:, num_vars_]) ** 2) /
     sum((X_new[:, num_vars_]) ** 2)

    For categorical variables(cat_vars_), the difference is defined as follows:

    sum(X_new[:, cat_vars_] != X_old[:, cat_vars_])) / n_cat_missing

    where X_new is the newly imputed array, X_old is the array imputed in the
    previous round, n_cat_missing is the total number of categorical
    values that are missing, and the sum() is performed both across rows
    and columns. Following [1], the stopping criterion is considered to have
    been met when difference between X_new and X_old increases for the first
    time for both types of variables (if available).

    Parameters
    ----------
    NOTE: Most parameter definitions below are taken verbatim from the
    Scikit-Learn documentation at [2] and [3].

    max_iter : int, optional (default = 10)
        The maximum iterations of the imputation process. Each column with a
        missing value is imputed exactly once in a given iteration.

    decreasing : boolean, optional (default = False)
        If set to True, columns are sorted according to decreasing number of
        missing values. In other words, imputation will move from imputing
        columns with the largest number of missing values to columns with
        fewest number of missing values.

    missing_values : np.nan, integer, optional (default = np.nan)
        The placeholder for the missing values. All occurrences of
        `missing_values` will be imputed.

    copy : boolean, optional (default = True)
        If True, a copy of X will be created. If False, imputation will
        be done in-place whenever possible.

    criterion : tuple, optional (default = ('squared_error', 'gini'))
        The function to measure the quality of a split.The first element of
        the tuple is for the Random Forest Regressor (for imputing numerical
        variables) while the second element is for the Random Forest
        Classifier (for imputing categorical variables).

    n_estimators : integer, optional (default=100)
        The number of trees in the forest.

    max_depth : integer or None, optional (default=None)
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int, float, optional (default=2)
        The minimum number of samples required to split an internal node:
        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

    min_samples_leaf : int, float, optional (default=1)
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.  This may have the effect of smoothing the model,
        especially in regression.
        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

    min_weight_fraction_leaf : float, optional (default=0.)
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    max_features : int, float, string or None, optional (default="auto")
        The number of features to consider when looking for the best split:
        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and
          `int(max_features * n_features)` features are considered at each
          split.
        - If "auto", then `max_features=sqrt(n_features)`.
        - If "sqrt", then `max_features=sqrt(n_features)` (same as "auto").
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.
        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    max_leaf_nodes : int or None, optional (default=None)
        Grow trees with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    min_impurity_decrease : float, optional (default=0.)
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.
        The weighted impurity decrease equation is the following::
            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)
        where ``N`` is the total number of samples, ``N_t`` is the number of
        samples at the current node, ``N_t_L`` is the number of samples in the
        left child, and ``N_t_R`` is the number of samples in the right child.
        ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
        if ``sample_weight`` is passed.

    bootstrap : boolean, optional (default=True)
        Whether bootstrap samples are used when building trees.

    oob_score : bool (default=False)
        Whether to use out-of-bag samples to estimate
        the generalization accuracy.

    n_jobs : int or None, optional (default=None)
        The number of jobs to run in parallel for both `fit` and `predict`.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    verbose : int, optional (default=0)
        Controls the verbosity when fitting and predicting.

    warm_start : bool, optional (default=False)
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit a whole
        new forest. See :term:`the Glossary <warm_start>`.

    class_weight : dict, list of dicts, "balanced", "balanced_subsample" or \
    None, optional (default=None)
        Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have weight one. For
        multi-output problems, a list of dicts can be provided in the same
        order as the columns of y.
        Note that for multioutput (including multilabel) weights should be
        defined for each class of every column in its own dict. For example,
        for four-class multilabel classification weights should be
        [{0: 1, 1: 1}, {0: 1, 1: 5}, {0: 1, 1: 1}, {0: 1, 1: 1}] instead of
        [{1:1}, {2:5}, {3:1}, {4:1}].
        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``
        The "balanced_subsample" mode is the same as "balanced" except that
        weights are computed based on the bootstrap sample for every tree
        grown.
        For multi-output, the weights of each column of y will be multiplied.
        Note that these weights will be multiplied with sample_weight (passed
        through the fit method) if sample_weight is specified.
        NOTE: This parameter is only applicable for Random Forest Classifier
        objects (i.e., for categorical variables).

    Attributes
    ----------
    statistics_ : Dictionary of length two
        The first element is an array with the mean of each numerical feature
        being imputed while the second element is an array of modes of
        categorical features being imputed (if available, otherwise it
        will be None).

    References
    ----------
    * [1] Stekhoven, Daniel J., and Peter Bühlmann. "MissForest—non-parametric
      missing value imputation for mixed-type data." Bioinformatics 28.1
      (2011): 112-118.
    * [2] https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.
      RandomForestRegressor.html#sklearn.ensemble.RandomForestRegressor
    * [3] https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.
      RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier

    Examples
    --------
    >>> from missingpy import MissForest
    >>> nan = float("NaN")
    >>> X = [[1, 2, nan], [3, 4, 3], [nan, 6, 5], [8, 8, 7]]
    >>> imputer = MissForest(random_state=1337)
    >>> imputer.fit_transform(X)
    Iteration: 0
    Iteration: 1
    Iteration: 2
    array([[1.  , 2. , 3.92 ],
           [3.  , 4. , 3. ],
           [2.71, 6. , 5. ],
           [8.  , 8. , 7. ]])
    """

    def __init__(self, seed=None, hyperparams=None, 
                 missing_values=np.nan, copy=True,
                 decreasing=False, max_iter=10,
                 n_jobs=-1, verbose=0):
        super().__init__(seed=seed)
        self.verbose = verbose
        self.missing_values = missing_values
        self.copy = copy
        self.n_jobs = n_jobs
        self.decreasing = decreasing
        self.max_iter = max_iter
        self.hyperparams = hyperparams
        self._predictors = defaultdict(dict)
        self._predictors_params = defaultdict(dict)
        
        if hyperparams is not None:
            print("Hyperparameters are provided and not be tuned.")
            self.classifier = RandomForestClassifier(random_state=seed, **hyperparams['RandomForestClassifier'])
            self.regressor = RandomForestRegressor(random_state=seed, **hyperparams['RandomForestRegressor'])
        else:
            print("Hyperparameters are not provided and will be tuned.")
            params_grid = get_missforest_params_for_tuning(seed)
            self.classifier_params_grid = params_grid['RandomForestClassifier']['params']
            self.regressor_params_grid = params_grid['RandomForestRegressor']['params']
            
            self.classifier_grid_search = GridSearchCV(
                            estimator=params_grid['RandomForestClassifier']['model'],
                            param_grid=self.classifier_params_grid,
                            scoring="f1_macro",
                            n_jobs=self.n_jobs,
                            cv=3, verbose=self.verbose)
            
            self.regressor_grid_search = GridSearchCV(
                            estimator=params_grid['RandomForestRegressor']['model'],
                            param_grid=self.regressor_params_grid,
                            scoring="neg_root_mean_squared_error",
                            n_jobs=self.n_jobs,
                            cv=3, verbose=self.verbose)
        
        
    def _miss_forest(self, Ximp, mask):
        """The missForest algorithm"""

        # Count missing per column
        col_missing_count = mask.sum(axis=0)

        # Get col and row indices for missing
        missing_rows, missing_cols = np.where(mask)

        if self.num_vars_ is not None:
            # Only keep indices for numerical vars
            keep_idx_num = np.in1d(missing_cols, self.num_vars_)
            missing_num_rows = missing_rows[keep_idx_num]
            missing_num_cols = missing_cols[keep_idx_num]

            # Make initial guess for missing values
            col_means = np.full(Ximp.shape[1], fill_value=np.nan)
            col_means[self.num_vars_] = self.statistics_.get('col_means')
            Ximp[missing_num_rows, missing_num_cols] = np.take(
                col_means, missing_num_cols)

        # If needed, repeat for categorical variables
        if self.cat_vars_ is not None:
            # Calculate total number of missing categorical values (used later)
            n_catmissing = np.sum(mask[:, self.cat_vars_])

            # Only keep indices for categorical vars
            keep_idx_cat = np.in1d(missing_cols, self.cat_vars_)
            missing_cat_rows = missing_rows[keep_idx_cat]
            missing_cat_cols = missing_cols[keep_idx_cat]

            # Make initial guess for missing values
            col_modes = np.full(Ximp.shape[1], fill_value=np.nan)
            col_modes[self.cat_vars_] = self.statistics_.get('col_modes')
            Ximp[missing_cat_rows, missing_cat_cols] = np.take(col_modes, missing_cat_cols)

        # 2. misscount_idx: sorted indices of cols in X based on missing count
        misscount_idx = np.argsort(col_missing_count)
        # Reverse order if decreasing is set to True
        if self.decreasing is True:
            misscount_idx = misscount_idx[::-1]
            
        self.misscount_idx = misscount_idx

        # 3. While new_gammas < old_gammas & self.iter_count_ < max_iter loop:
        self.iter_count_ = 0
        gamma_new = 0
        gamma_old = np.inf
        gamma_newcat = 0
        gamma_oldcat = np.inf
        self.col_index = np.arange(Ximp.shape[1])

        while ((gamma_new < gamma_old) or (gamma_newcat < gamma_oldcat)) and \
                (self.iter_count_ < self.max_iter):

            # 4. store previously imputed matrix
            Ximp_old = np.copy(Ximp)
            if self.iter_count_ != 0:
                gamma_old = gamma_new
                gamma_oldcat = gamma_newcat
            # 5. loop
            for s in self.misscount_idx:
                # Column indices other than the one being imputed
                s_prime = np.delete(self.col_index, s)

                # Get indices of rows where 's' is observed and missing
                obs_rows = np.where(~mask[:, s])[0]
                mis_rows = np.where(mask[:, s])[0]

                # If no missing, then skip
                if len(mis_rows) == 0:
                    continue

                # Get observed values of 's'
                yobs = Ximp[obs_rows, s]

                # Get 'X' for both observed and missing 's' column
                xobs = Ximp[np.ix_(obs_rows, s_prime)]
                xmis = Ximp[np.ix_(mis_rows, s_prime)]

                # 6. Fit a random forest over observed and predict the missing
                if self.cat_vars_ is not None and s in self.cat_vars_:
                    if self.hyperparams is not None:
                        self.classifier.fit(X=xobs, y=yobs)
                        # 7. predict ymis(s) using xmis(x)
                        ymis = self.classifier.predict(xmis)
                        # 8. update imputed matrix using predicted matrix ymis(s)
                        Ximp[mis_rows, s] = ymis
                        # save the predictor
                        self._predictors[s][self.iter_count_] = self.classifier
                    else:
                        self.classifier_grid_search.fit(X=xobs, y=yobs)
                        best_classifier = self.classifier_grid_search.best_estimator_
                        best_classifier.fit(X=xobs, y=yobs)
                        # 7. predict ymis(s) using xmis(x)
                        ymis = best_classifier.predict(xmis)
                        # 8. update imputed matrix using predicted matrix ymis(s)
                        Ximp[mis_rows, s] = ymis
                        # save the predictor
                        self._predictors[s][self.iter_count_] = best_classifier
                        self._predictors_params[s][self.iter_count_] = self.classifier_grid_search.best_params_
                else:
                    if self.hyperparams is not None:
                        self.regressor.fit(X=xobs, y=yobs)
                        # 7. predict ymis(s) using xmis(x)
                        ymis = self.regressor.predict(xmis)
                        # 8. update imputed matrix using predicted matrix ymis(s)
                        Ximp[mis_rows, s] = ymis
                        # save the predictor
                        self._predictors[s][self.iter_count_] = self.regressor
                    else:
                        self.regressor_grid_search.fit(X=xobs, y=yobs)
                        best_regressor = self.regressor_grid_search.best_estimator_
                        best_regressor.fit(X=xobs, y=yobs)
                        # 7. predict ymis(s) using xmis(x)
                        ymis = best_regressor.predict(xmis)
                        # 8. update imputed matrix using predicted matrix ymis(s)
                        Ximp[mis_rows, s] = ymis
                        # save the predictor
                        self._predictors[s][self.iter_count_] = best_regressor
                        self._predictors_params[s][self.iter_count_] = self.regressor_grid_search.best_params_

            # 9. Update gamma (stopping criterion)
            if self.cat_vars_ is not None:
                gamma_newcat = np.sum(
                    (Ximp[:, self.cat_vars_] != Ximp_old[:, self.cat_vars_])) / n_catmissing
            if self.num_vars_ is not None:
                gamma_new = np.sum((Ximp[:, self.num_vars_] - Ximp_old[:, self.num_vars_]) ** 2) / np.sum((Ximp[:, self.num_vars_]) ** 2)

            if self.verbose:
                print("="*100)
                print("MissForestImputer Iteration:", self.iter_count_)
                gamma_output = f"Gamma_numerical_new = {gamma_new:.5f}, Gamma_numerical_old = {gamma_old:.5f}\n" if self.num_vars_ is not None else ""
                gamma_output += f"Gamma_categorical_new = {gamma_newcat:.5f}, Gamma_categorical_old = {gamma_oldcat:.5f}\n" if self.cat_vars_ is not None else ""
            self.iter_count_ += 1

        return Ximp_old

    def _validate_input(self, df):
        # Check data integrity and calling arguments
        force_all_finite = False if self.missing_values in ["NaN",
                                                            np.nan] else True

        X = check_array(df, accept_sparse=False, dtype=np.float64,
                        force_all_finite=force_all_finite, copy=self.copy)

        # Check for +/- inf
        if np.any(np.isinf(X)):
            raise ValueError("+/- inf values are not supported.")

        # Check if any column has all missing
        mask = _get_mask(X, self.missing_values)
        if np.any(mask.sum(axis=0) >= (X.shape[0])):
            raise ValueError("One or more columns have all rows missing.")
        
        return X, mask

    def fit(self, X, y=None, cat_vars=None):
        """Fit the imputer on X.

        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            Input data, where ``n_samples`` is the number of samples and
            ``n_features`` is the number of features.

        cat_vars : int or array of ints, optional (default = None)
            An int or an array containing column indices of categorical
            variable(s)/feature(s) present in the dataset X.
            ``None`` if there are no categorical variables in the dataset.

        Returns
        -------
        self : object
            Returns self.
        """

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

        # First replace missing values with NaN if it is something else
        if self.missing_values not in ['NaN', np.nan]:
            X[np.where(X == self.missing_values)] = np.nan

        # Now, make initial guess for missing values
        col_means = np.nanmean(X[:, num_vars], axis=0) if num_vars is not None else None
        col_modes = mode(
            X[:, cat_vars], axis=0, nan_policy='omit')[0] if cat_vars is not \
                                                           None else None

        self.cat_vars_ = cat_vars
        self.num_vars_ = num_vars
        self.statistics_ = {"col_means": col_means, "col_modes": col_modes}
        
        self._miss_forest(X, mask)

        return self

    def transform(self, X):
        """Impute all missing values in X.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            The input data to complete.

        Returns
        -------
        X : {array-like}, shape = [n_samples, n_features]
            The imputed dataset.
        """
        # Confirm whether fit() has been called
        check_is_fitted(self, ["cat_vars_", "num_vars_", "statistics_"])

        X, mask = self._validate_input(X)

        # Get fitted X col count and ensure correct dimension
        n_cols_fit_X = (0 if self.num_vars_ is None else len(self.num_vars_)) \
            + (0 if self.cat_vars_ is None else len(self.cat_vars_))
        _, n_cols_X = X.shape

        if n_cols_X != n_cols_fit_X:
            raise ValueError("Incompatible dimension between the fitted "
                             "dataset and the one to be transformed.")

        # Check if anything is actually missing and if not return original X                             
        mask = _get_mask(X, self.missing_values)
        if not mask.sum() > 0:
            warnings.warn("No missing value located; returning original "
                          "dataset.")
            return X
        
        # Get col and row indices for missing
        missing_rows, missing_cols = np.where(mask)

        if self.num_vars_ is not None:
            # Only keep indices for numerical vars
            keep_idx_num = np.in1d(missing_cols, self.num_vars_)
            missing_num_rows = missing_rows[keep_idx_num]
            missing_num_cols = missing_cols[keep_idx_num]

            # Make initial guess for missing values
            col_means = np.full(X.shape[1], fill_value=np.nan)
            col_means[self.num_vars_] = self.statistics_.get('col_means')
            X[missing_num_rows, missing_num_cols] = np.take(
                col_means, missing_num_cols)
            
        if self.cat_vars_ is not None:
            # Only keep indices for categorical vars
            keep_idx_cat = np.in1d(missing_cols, self.cat_vars_)
            missing_cat_rows = missing_rows[keep_idx_cat]
            missing_cat_cols = missing_cols[keep_idx_cat]

            # Make initial guess for missing values
            col_modes = np.full(X.shape[1], fill_value=np.nan)
            col_modes[self.cat_vars_] = self.statistics_.get('col_modes')
            X[missing_cat_rows, missing_cat_cols] = np.take(col_modes, missing_cat_cols)

        for i in range(self.iter_count_):
            for s in self.misscount_idx:
                s_prime = np.delete(self.col_index, s)
                mis_rows = np.where(mask[:, s])[0]

                # If no missing, then skip
                if len(mis_rows) == 0:
                    continue

                xmis = X[np.ix_(mis_rows, s_prime)]
                
                ymis = self._predictors[s][i].predict(xmis)
                X[mis_rows, s] = ymis

        # Return imputed dataset
        return X

    def fit_transform(self, X, y=None, **fit_params):
        """Fit MissForest and impute all missing values in X.

        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            Input data, where ``n_samples`` is the number of samples and
            ``n_features`` is the number of features.

        Returns
        -------
        X : {array-like}, shape (n_samples, n_features)
            Returns imputed dataset.
        """
        return self.fit(X, **fit_params).transform(X)
    
    def get_predictors_params(self):
        if self.hyperparams is None:
            return self._predictors_params
        
        return self.hyperparams
    