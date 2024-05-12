import scipy
import numpy as np
import pandas as pd
from scipy.stats import entropy, gaussian_kde
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

from source.preprocessing import get_simple_preprocessor


def preprocess_base_flow_dataset(base_flow_dataset):
    column_transformer = get_simple_preprocessor(base_flow_dataset)
    column_transformer = column_transformer.set_output(transform="pandas")  # Set transformer output to a pandas df
    base_flow_dataset.X_train_val = column_transformer.fit_transform(base_flow_dataset.X_train_val)
    base_flow_dataset.X_test = column_transformer.transform(base_flow_dataset.X_test)

    return base_flow_dataset


def preprocess_mult_base_flow_datasets(main_base_flow_dataset, extra_base_flow_datasets):
    column_transformer = get_simple_preprocessor(main_base_flow_dataset)
    column_transformer = column_transformer.set_output(transform="pandas")  # Set transformer output to a pandas df

    # Preprocess main_base_flow_dataset
    main_base_flow_dataset.X_train_val = column_transformer.fit_transform(main_base_flow_dataset.X_train_val)
    main_base_flow_dataset.X_test = column_transformer.transform(main_base_flow_dataset.X_test)

    # Preprocess extra_base_flow_datasets
    extra_test_sets = []
    for i in range(len(extra_base_flow_datasets)):
        extra_base_flow_datasets[i].X_test = column_transformer.transform(extra_base_flow_datasets[i].X_test)
        extra_test_sets.append((extra_base_flow_datasets[i].X_test,
                                extra_base_flow_datasets[i].y_test,
                                extra_base_flow_datasets[i].init_sensitive_attrs_df))

    return main_base_flow_dataset, extra_test_sets


def get_object_columns_indexes(df):
    """
    Get the indexes of columns with object dtype in a pandas DataFrame.

    Parameters:
    df (pd.DataFrame): Input pandas DataFrame.

    Returns:
    list: Indexes of columns with object dtype.
    """
    object_columns = df.select_dtypes(include=['object']).columns
    object_indexes = [df.columns.get_loc(col) for col in object_columns]
    
    return object_indexes


def encode_cat(X_c):
    data = X_c.copy()
    nonulls = data.dropna().values
    impute_reshape = nonulls.reshape(-1,1)
    encoder = OrdinalEncoder()
    impute_ordinal = encoder.fit_transform(impute_reshape)
    data.loc[data.notnull()] = np.squeeze(impute_ordinal)

    return data, encoder


def encode_cat_with_existing_encoder(X_c, encoder):
    data = X_c.copy()
    nonulls = data.dropna().values
    impute_reshape = nonulls.reshape(-1,1)
    impute_ordinal = encoder.transform(impute_reshape)
    data.loc[data.notnull()] = np.squeeze(impute_ordinal)

    return data


def decode_cat(X_c, encoder):
    data = X_c.copy()
    nonulls = data.dropna().values.reshape(-1,1)
    n_cat = len(encoder.categories_[0])
    nonulls = np.round(nonulls).clip(0, n_cat-1)
    nonulls = encoder.inverse_transform(nonulls)
    data.loc[data.notnull()] = np.squeeze(nonulls)

    return data


def get_numerical_columns_indexes(df):
    """
    Get the indexes of columns with numerical dtype in a pandas DataFrame.

    Parameters:
    df (pd.DataFrame): Input pandas DataFrame.

    Returns:
    list: Indexes of columns with numerical dtype.
    """
    numerical_columns = df.select_dtypes(include=['int', 'float']).columns
    numerical_indexes = [df.columns.get_loc(col) for col in numerical_columns]
    
    return numerical_indexes


def _get_mask(X, value_to_mask):
    """Compute the boolean mask X == missing_values."""
    if value_to_mask == "NaN" or np.isnan(value_to_mask):
        return np.isnan(X)
    else:
        return X == value_to_mask


def get_columns_sorted_by_nulls(mask):
    # Calculate the number of null values in each column
    null_counts = mask.sum()

    # Sort columns based on the number of null values
    sorted_columns = null_counts.sort_values(ascending=True)

    # Get the column names as a list
    sorted_columns_names = sorted_columns.index.tolist()

    return sorted_columns_names


def calculate_kl_divergence_with_histograms(true: pd.DataFrame, pred: pd.DataFrame, verbose: bool = False):
    if verbose:
        print('Compute KL divergence using histograms...')

    # Get the value counts normalized to probability distributions
    true_dist = true.value_counts(normalize=True)
    pred_dist = pred.value_counts(normalize=True)

    # Ensure both distributions have the same index
    all_categories = true_dist.index.union(pred_dist.index)
    true_dist = true_dist.reindex(all_categories, fill_value=0.000000001).sort_index()
    pred_dist = pred_dist.reindex(all_categories, fill_value=0.000000001).sort_index()

    # Calculate KL divergence from true_dist to pred_dist
    # KL(P || Q) where P is the true distribution and Q is the approximation
    kl_div = entropy(true_dist, pred_dist)

    return kl_div


def calculate_kl_divergence_with_kde(true: pd.DataFrame, pred: pd.DataFrame, verbose: bool = False):
    # Normalize true and pred series
    scaler = StandardScaler().set_output(transform="pandas")
    true_scaled = scaler.fit_transform(true.to_frame())
    true_scaled = true_scaled[true_scaled.columns[0]]
    pred_scaled = scaler.transform(pred.to_frame())
    pred_scaled = pred_scaled[pred_scaled.columns[0]]

    if true.shape[0] <= 1:
        # In case when subgroup true df has only one sample, return None
        return None
    elif pred.nunique() == 1:
        if verbose:
            print('Compute KL divergence using KDE and discrete uniform PMF...')

        # Estimate probability density functions using kernel density estimation
        true_kde = gaussian_kde(true_scaled)

        # Create the discrete uniform distribution with one value
        discrete_uniform_values = [pred_scaled.values[0]]
        discrete_uniform_pmf = [1.0]  # Probability mass function for the single value
        pred_kde = scipy.stats.rv_discrete(values=(discrete_uniform_values, discrete_uniform_pmf))

        # Compute the PMF/PDF for both distributions
        x = np.linspace(min(min(true_scaled), min(pred_scaled)), max(max(true_scaled), max(pred_scaled)), 999)
        x = np.append(x, discrete_uniform_values)

        true_dist = true_kde.evaluate(x)
        pred_dist = pred_kde.pmf(x)
        pred_dist[pred_dist == 0.] = 0.000000001  # replace zeros to avoid infinity in scipy.entropy

    else:
        if verbose:
            print('Compute KL divergence using KDE...')

        # Estimate probability density functions using kernel density estimation
        true_kde = gaussian_kde(true_scaled)
        pred_kde = gaussian_kde(pred_scaled)

        # Evaluate KDEs at a set of points
        x = np.linspace(min(min(true_scaled), min(pred_scaled)), max(max(true_scaled), max(pred_scaled)), 1000)
        true_dist = true_kde.evaluate(x)
        pred_dist = pred_kde.evaluate(x)
        pred_dist[pred_dist == 0.] = 0.000000001  # replace zeros to avoid infinity in scipy.entropy

    # Calculate KL divergence from true_dist to pred_dist
    # KL(P || Q) where P is the true distribution and Q is the approximation
    return entropy(true_dist, pred_dist)


def calculate_kl_divergence(true: pd.DataFrame, pred: pd.DataFrame, column_type: str, verbose: bool = False):
    if column_type == 'numerical':
        # Compute KL divergence for numerical features
        kl_div = calculate_kl_divergence_with_kde(true, pred, verbose=verbose)
    else:
        # Compute KL divergence for categorical features
        kl_div = calculate_kl_divergence_with_histograms(true, pred, verbose=verbose)

    return kl_div
