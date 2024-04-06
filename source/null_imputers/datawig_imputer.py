import os
import shutil
import logging
import pandas as pd


def complete(X_train_with_nulls: pd.DataFrame,
             X_test_with_nulls: pd.DataFrame,
             numeric_columns_with_nulls: list,
             categorical_columns_with_nulls: list,
             precision_threshold: float = 0.0,
             hpo: bool = False,
             num_epochs: int = 100,
             iterations: int = 1,
             output_path: str = "."):
    """
    Given a dataframe with missing values, this function detects all imputable columns, trains an imputation model
    on all other columns and imputes values for each missing value.
    Several imputation iterators can be run.
    Imputable columns are either numeric columns or non-numeric categorical columns; for determining whether a
        column is categorical (as opposed to a plain text column) we use the following heuristic:
        a non-numeric categorical column should have least 10 times as many rows as there were unique values
    If an imputation model did not reach the precision specified in the precision_threshold parameter for a given
        imputation value, that value will not be imputed; thus depending on the precision_threshold, the returned
        dataframe can still contain some missing values.
    For numeric columns, we do not filter for accuracy.
    :param data_frame: original dataframe
    :param precision_threshold: precision threshold for categorical imputations (default: 0.0)
    :param inplace: whether or not to perform imputations inplace (default: False)
    :param hpo: whether or not to perform hyperparameter optimization (default: False)
    :param verbose: verbosity level, values > 0 log to stdout (default: 0)
    :param num_epochs: number of epochs for each imputation model training (default: 100)
    :param iterations: number of iterations for iterative imputation (default: 1)
    :param output_path: path to store model and metrics

    :return: dataframe with imputations
    """
    # Import datawig inside a function to avoid its installation to use other null imputers
    import datawig

    os.environ['MXNET_LOG_LEVEL'] = 'ERROR'
    os.environ['MXNET_STORAGE_FALLBACK_LOG_VERBOSE'] = '0'

    train_missing_mask = X_train_with_nulls.copy().isnull()
    test_missing_mask = X_test_with_nulls.copy().isnull()
    X_train_imputed = X_train_with_nulls.copy()
    X_test_imputed = X_test_with_nulls.copy()

    col_set = set(X_train_imputed.columns)
    null_imputer_params = dict()
    for _ in range(iterations):
        for output_col in set(numeric_columns_with_nulls) | set(categorical_columns_with_nulls):
            # Reset logger handler
            if datawig.utils.logger.hasHandlers():
                datawig.utils.logger.handlers.clear()
            datawig.utils.logger.addHandler(datawig.utils.consoleHandler)
            datawig.utils.set_stream_log_level(logging.DEBUG)

            datawig.utils.logger.info(f'Start null imputation for the {output_col} column')

            # train on all input columns but the to-be-imputed one
            input_cols = list(col_set - set([output_col]))

            # train on all observed values
            train_idx_missing = train_missing_mask[output_col]

            imputer = datawig.SimpleImputer(input_columns=input_cols,
                                            output_column=output_col,
                                            output_path=os.path.join(output_path, output_col))
            if hpo:
                imputer.fit_hpo(X_train_imputed.loc[~train_idx_missing, :],
                                patience=5,
                                num_epochs=1,
                                # num_epochs=100,
                                final_fc_hidden_units=[[10]])
                                # final_fc_hidden_units=[[10], [50], [100]])
            else:
                imputer.fit(X_train_imputed.loc[~train_idx_missing, :],
                            patience=5,
                            num_epochs=num_epochs,
                            batch_size=64,
                            calibrate=False)

            print('output_col: ', output_col, flush=True)
            print('imputer.output_type: ', imputer.output_type, flush=True)
            print('imputer.numeric_columns: ', imputer.numeric_columns, flush=True)
            print('imputer.string_columns: ', imputer.string_columns, flush=True)

            tmp_train = imputer.predict(X_train_imputed, precision_threshold=precision_threshold)
            X_train_imputed.loc[train_idx_missing, output_col] = tmp_train[output_col + "_imputed"]

            test_idx_missing = test_missing_mask[output_col]
            tmp_test = imputer.predict(X_test_imputed, precision_threshold=precision_threshold)
            X_test_imputed.loc[test_idx_missing, output_col] = tmp_test[output_col + "_imputed"]

            # Select hyper-params of the best model
            if imputer.hpo.results.shape[0] == 0:
                null_imputer_params[output_col] = None
            else:
                if imputer.output_type == 'numeric':
                    best_model_idx = imputer.hpo.results['mse'].astype(float).idxmin()
                else:
                    best_model_idx = imputer.hpo.results['precision_weighted'].astype(float).idxmax()

                print('best_model_idx -- ', best_model_idx, flush=True)
                best_model_idx = int(best_model_idx)
                null_imputer_params[output_col] = str(imputer.hpo.results.iloc[best_model_idx].to_dict())

            # remove the directory with logfiles for this column
            shutil.rmtree(os.path.join(output_path, output_col))

            datawig.utils.logger.info(f'Successfully completed null imputation for the {output_col} column')

    return X_train_imputed, X_test_imputed, null_imputer_params
