import os
import shutil
import pandas as pd


def complete(data_frame: pd.DataFrame,
             precision_threshold: float = 0.0,
             inplace: bool = False,
             hpo: bool = False,
             verbose: int = 0,
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
    missing_mask = data_frame.copy().isnull()

    if inplace is False:
        data_frame = data_frame.copy()

    numeric_columns = [c for c in data_frame.columns if is_numeric_dtype(data_frame[c])]
    string_columns = list(set(data_frame.columns) - set(numeric_columns))
    logger.debug("Assuming numerical columns: {}".format(", ".join(numeric_columns)))

    col_set = set(numeric_columns + string_columns)

    categorical_columns = [col for col in string_columns if SimpleImputer._is_categorical(data_frame[col])]
    logger.debug("Assuming categorical columns: {}".format(", ".join(categorical_columns)))
    for _ in range(iterations):
        for output_col in set(numeric_columns) | set(categorical_columns):
            # train on all input columns but the to-be-imputed one
            input_cols = list(col_set - set([output_col]))

            # train on all observed values
            idx_missing = missing_mask[output_col]

            imputer = SimpleImputer(input_columns=input_cols,
                                    output_column=output_col,
                                    output_path=os.path.join(output_path, output_col))
            if hpo:
                imputer.fit_hpo(data_frame.loc[~idx_missing, :],
                                patience=5 if output_col in categorical_columns else 20,
                                num_epochs=100,
                                final_fc_hidden_units=[[0], [10], [50], [100]])
            else:
                imputer.fit(data_frame.loc[~idx_missing, :],
                            patience=5 if output_col in categorical_columns else 20,
                            num_epochs=num_epochs,
                            calibrate=False)

            tmp = imputer.predict(data_frame, precision_threshold=precision_threshold)
            data_frame.loc[idx_missing, output_col] = tmp[output_col + "_imputed"]

            # remove the directory with logfiles for this column
            shutil.rmtree(os.path.join(output_path, output_col))

    return data_frame
