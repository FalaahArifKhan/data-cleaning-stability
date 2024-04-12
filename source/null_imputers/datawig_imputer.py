import os
import uuid
import shutil
import logging
import pandas as pd

from datetime import datetime
from datawig.utils import logger


def complete(X_train_with_nulls: pd.DataFrame,
             X_test_with_nulls: pd.DataFrame,
             numeric_columns_with_nulls: list,
             categorical_columns_with_nulls: list,
             all_numeric_columns: list,
             all_categorical_columns: list,
             hyperparams: dict,
             output_path: str = ".",
             **kwargs):
    # Import datawig inside a function to avoid its installation to use other null imputers
    import datawig

    os.environ['MXNET_LOG_LEVEL'] = 'ERROR'
    os.environ['MXNET_STORAGE_FALLBACK_LOG_VERBOSE'] = '0'

    precision_threshold = kwargs['precision_threshold']
    num_epochs = kwargs['num_epochs']
    iterations = kwargs['iterations']

    train_missing_mask = X_train_with_nulls.copy().isnull()
    test_missing_mask = X_test_with_nulls.copy().isnull()
    X_train_imputed = X_train_with_nulls.copy()
    X_test_imputed = X_test_with_nulls.copy()

    # Define column types for each feature column in X dataframe
    hps = dict()
    for numeric_column_name in all_numeric_columns:
        hps[numeric_column_name] = dict()
        hps[numeric_column_name]['type'] = ['numeric']

    for categorical_column_name in all_categorical_columns:
        hps[categorical_column_name] = dict()
        hps[categorical_column_name]['type'] = ['categorical']

    col_set = set(X_train_imputed.columns)
    null_imputer_params_dct = dict()
    for _ in range(iterations):
        for output_col in set(numeric_columns_with_nulls) | set(categorical_columns_with_nulls):
            datetime_now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            column_output_path = os.path.join(output_path, f'{output_col}_{datetime_now_str}_{str(uuid.uuid1())}')

            # Reset logger handler
            if datawig.utils.logger.hasHandlers():
                datawig.utils.logger.handlers.clear()
            datawig.utils.logger.addHandler(datawig.utils.consoleHandler)
            datawig.utils.set_stream_log_level(logging.ERROR)

            datawig.utils.logger.info(f'Start null imputation for the {output_col} column')

            # train on all input columns but the to-be-imputed one
            input_cols = list(col_set - set([output_col]))

            # train on all observed values
            train_idx_missing = train_missing_mask[output_col]

            imputer = datawig.SimpleImputer(input_columns=input_cols,
                                            output_column=output_col,
                                            output_path=column_output_path)
            if hyperparams is None:
                imputer.fit_hpo(X_train_imputed.loc[~train_idx_missing, :],
                                hps=hps,
                                num_evals=6,
                                patience=3,
                                num_epochs=num_epochs,
                                batch_size=64,
                                final_fc_hidden_units=[[10], [50], [100]])
            else:
                imputer.fit(X_train_imputed.loc[~train_idx_missing, :],
                            final_fc_hidden_units=hyperparams['final_fc_hidden_units'],
                            patience=3,
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
                null_imputer_params_dct[output_col] = None
            else:
                if imputer.output_type == 'numeric':
                    best_imputer_idx = imputer.hpo.results['mse'].astype(float).idxmin()
                else:
                    best_imputer_idx = imputer.hpo.results['precision_weighted'].astype(float).idxmax()

                best_imputer_idx = int(best_imputer_idx)
                null_imputer_params = imputer.hpo.results.iloc[best_imputer_idx].to_dict()
                null_imputer_params['best_imputer_idx'] = best_imputer_idx
                null_imputer_params_dct[output_col] = null_imputer_params

            # remove the directory with logfiles for this column
            shutil.rmtree(column_output_path)

            datawig.utils.logger.info(f'Successfully completed null imputation for the {output_col} column')

    return X_train_imputed, X_test_imputed, null_imputer_params_dct
