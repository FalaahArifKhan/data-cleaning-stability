"""
The original paper for the code below:
- GitHub: https://github.com/guaiyoui/NOMI

- Citation:
@article{wang2024uncertainty,
  title={Missing Data Imputation with Uncertainty-Driven Network},
  author={Wang, Jianwei and Zhang, Ying and Wang, Kai and Lin, Xuemin and Zhang, Wenjie},
  journal={Proceedings of the ACM on Management of Data},
  volume={2},
  number={3},
  pages={1--25},
  year={2024},
  publisher={ACM New York, NY, USA}
}
"""
import torch
import numpy as np
import hnswlib
import tensorflow as tf
from distutils.version import LooseVersion
from tqdm import tqdm

# Check if the required dependencies are available
if LooseVersion(tf.__version__) >= LooseVersion("2.16"):
    import neural_tangents as nt
    from neural_tangents import stax


def sample_batch_index(total, batch_size):
    """
    Sample index of the mini-batch.

    Args:
      - total: total number of samples
      - batch_size: batch size

    Returns:
      - batch_idx: batch index
    """
    total_idx = np.random.permutation(total)
    batch_idx = total_idx[:batch_size]
    return batch_idx


def dist2sim(neigh_dist):
    if torch.is_tensor(neigh_dist):
        neigh_dist = neigh_dist.cpu().detach().numpy()
    with np.errstate(divide="ignore"):
        dist = 1.0 / neigh_dist

    inf_mask = np.isinf(dist)
    inf_row = np.any(inf_mask, axis=1)
    dist[inf_row] = inf_mask[inf_row]
    denom = np.sum(dist, axis=1)
    denom = denom.reshape((-1,1))

    return dist/denom


def prediction(pred_fn, X_test, kernel_type = "nngp", compute_cov = True):
    pred_mean, pred_cov = pred_fn(x_test=X_test, get=kernel_type, compute_cov=compute_cov)
    return pred_mean, pred_cov


def normalization_std(data):
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)

    normalized_data = (data - min_vals) / (max_vals - min_vals)
    return normalized_data + 1


def normalization(data, parameters=None):
    """
    Normalize data in [0, 1] range.

    Args:
      - data: original data

    Returns:
      - norm_data: normalized data
      - norm_parameters: min_val, max_val for each feature for renormalization
    """

    # Parameters
    _, dim = data.shape
    norm_data = data.copy()

    if parameters is None:

        # MixMax normalization
        min_val = np.zeros(dim)
        max_val = np.zeros(dim)

        # For each dimension
        for i in range(dim):
            min_val[i] = np.nanmin(norm_data[:,i])
            norm_data[:,i] = norm_data[:,i] - np.nanmin(norm_data[:,i])
            max_val[i] = np.nanmax(norm_data[:,i])
            norm_data[:,i] = norm_data[:,i] / (np.nanmax(norm_data[:,i]) + 1e-6)

        # Return norm_parameters for renormalization
        norm_parameters = {'min_val': min_val,
                           'max_val': max_val}

    else:
        min_val = parameters['min_val']
        max_val = parameters['max_val']

        # For each dimension
        for i in range(dim):
            norm_data[:,i] = norm_data[:,i] - min_val[i]
            norm_data[:,i] = norm_data[:,i] / (max_val[i] + 1e-6)

        norm_parameters = parameters

    return norm_data, norm_parameters


def renormalization(norm_data, norm_parameters):
    """
    Renormalize data from [0, 1] range to the original range.

    Args:
      - norm_data: normalized data
      - norm_parameters: min_val, max_val for each feature for renormalization

    Returns:
      - renorm_data: renormalized original data
    """

    min_val = norm_parameters['min_val']
    max_val = norm_parameters['max_val']

    _, dim = norm_data.shape
    renorm_data = norm_data.copy()

    for i in range(dim):
        renorm_data[:,i] = renorm_data[:,i] * (max_val[i] + 1e-6)
        renorm_data[:,i] = renorm_data[:,i] + min_val[i]

    return renorm_data


def rounding(imputed_data, cat_indices):
    """
    Round imputed data for categorical variables.

    Args:
      - imputed_data: imputed data
      - data_x: original data with missing values
      - cat_indices: indices of categorical columns

    Returns:
      - rounded_data: rounded imputed data
    """
    rounded_data = imputed_data.copy()
    for i in cat_indices:
        rounded_data[:, i] = np.round(rounded_data[:, i])

    return rounded_data


class NOMIImputer:
    def __init__(self, k_neighbors=10, similarity_metric="l2", max_iterations=3, tau=1.0, beta=1.0):
        self.k_neighbors = k_neighbors
        self.similarity_metric = similarity_metric
        self.max_iterations = max_iterations
        self.tau = tau
        self.beta = beta

        self.index_dct = dict()
        self.predict_fn_dct = dict()
        self.Y_train_dct = dict()
        self.is_fitted = False

    def fit_transform(self, X, num_indices_with_nulls, cat_indices_with_nulls):
        """
        Fit and transform with the NOMI imputer using the provided training data.

        Parameters:
        - X: numpy array of shape (n_samples, n_features)
            The dataset containing missing values.
        - num_indices_with_nulls: indices of numerical columns with nulls.
        - cat_indices_with_nulls: indices of categorical columns with nulls.

        Returns:
        - numpy array of shape (n_samples, n_features)
            Dataset with imputed values.
        """
        data_x = X
        data_m = 1 - np.isnan(data_x) # Reverse mask to comply with the NOMI API
        norm_data, norm_parameters = normalization(data_x)
        norm_data_x = np.nan_to_num(norm_data, 0)

        num, dims = norm_data_x.shape
        imputed_X = norm_data_x.copy()
        data_m_imputed = data_m.copy()
        col_indices_with_nulls = num_indices_with_nulls + cat_indices_with_nulls

        # Step 1: Model Initialization
        _, _, kernel_fn = stax.serial(
            stax.Dense(2 * dims), stax.Relu(),
            stax.Dense(dims), stax.Relu(),
            stax.Dense(1), stax.Sigmoid_like()
        )
        # Step 2: Iterative Imputation Process
        for iteration in range(self.max_iterations):
            print(f'Started iteration {iteration + 1}', flush=True)

            # Iterates over each dimension of the dataset
            for dim in tqdm(col_indices_with_nulls, desc="Training"):
                # Extract observed values
                X_wo_dim = np.delete(imputed_X, dim, 1)
                i_not_nan_index = data_m_imputed[:, dim].astype(bool)

                # Check for observed values
                if not np.any(i_not_nan_index):
                    print(f"No observed values for dimension {dim}, skipping.")
                    continue

                # Create training and test sets (X_train, Y_train for observed, X_test for missing)
                X_train = X_wo_dim[i_not_nan_index]
                Y_train = imputed_X[i_not_nan_index, dim]

                X_test = X_wo_dim[~i_not_nan_index]
                true_indices = np.where(~i_not_nan_index)[0]

                if X_test.shape[0] == 0:
                    continue

                no, d = X_train.shape

                # Use hnswlib.Index for nearest-neighbor search on X_train.
                # Select k neighbors (args.k_candidate) to calculate distances and weights.
                index = hnswlib.Index(space=self.similarity_metric, dim=d)
                index.init_index(max_elements=no, ef_construction=200, M=16)
                index.add_items(X_train)
                index.set_ef(int(self.k_neighbors * 1.2))

                if X_train.shape[0] > 300:
                    batch_idx = sample_batch_index(X_train.shape[0], 300)
                else:
                    batch_idx = sample_batch_index(X_train.shape[0], X_train.shape[0])

                X_batch = X_train[batch_idx,:]
                Y_batch = Y_train[batch_idx]

                neigh_ind, neigh_dist = index.knn_query(X_batch, k=self.k_neighbors, filter=None)
                neigh_dist = np.sqrt(neigh_dist)

                # Calculate weights (dist2sim) for neighbor contributions.
                weights = dist2sim(neigh_dist[:,1:])

                # Prepare train_input and test_input matrices by combining weights with neighbor values.
                y_neighbors = Y_train[neigh_ind[:,1:]]
                train_input = weights * y_neighbors

                neigh_ind_test, neigh_dist_test = index.knn_query(X_test, k=self.k_neighbors, filter=None)
                neigh_dist_test = np.sqrt(neigh_dist_test)

                weights_test = dist2sim(neigh_dist_test[:, :-1])
                y_neighbors_test = Y_train[neigh_ind_test[:, :-1]]
                test_input = weights_test * y_neighbors_test

                # Use a prediction function (nt.predict.gradient_descent_mse_ensemble)
                # to learn a regression model and predict missing values.
                predict_fn = nt.predict.gradient_descent_mse_ensemble(kernel_fn=kernel_fn,
                                                                      x_train=train_input,
                                                                      y_train=Y_batch.reshape(-1, 1),
                                                                      diag_reg=1e-4)

                y_pred, pred_cov = prediction(predict_fn, test_input, kernel_type="nngp")

                if iteration == 0:
                    # Replace missing values directly with predictions
                    imputed_X[~i_not_nan_index, dim] = y_pred.reshape(-1)
                elif iteration <= 3:
                    # Normalize uncertainty
                    pred_std = np.sqrt(np.diag(pred_cov))
                    pred_std = np.ravel(np.array(pred_std))
                    pred_std = normalization_std(pred_std)
                    pred_std = np.nan_to_num(pred_std, nan=1.0)

                    # Adjust imputed values using a weighted combination of prior and current predictions based on pred_std
                    greater_than_threshold_indices = np.where(pred_std <= self.tau)[0]

                    for i in range(greater_than_threshold_indices.shape[0]):
                        data_m_imputed[true_indices[greater_than_threshold_indices[i]]:, dim] = 1

                    imputed_X[~i_not_nan_index, dim] = (1 - self.beta / pred_std) * imputed_X[~i_not_nan_index, dim] + self.beta / pred_std * y_pred.reshape(-1)
                else:
                    imputed_X[~i_not_nan_index, dim] = y_pred.reshape(-1)

                # Save the latest fitted variables for the transform method.
                # In case of multiple iterations per dimension, these variables will be overriden.
                self.index_dct[dim] = index
                self.predict_fn_dct[dim] = predict_fn
                self.Y_train_dct[dim] = Y_train

            print(f'Finished iteration {iteration + 1}', flush=True)

        # Step 3: Post-Processing
        imputed_data = renormalization(imputed_X, norm_parameters) # Re-normalize the imputed data
        imputed_data = rounding(imputed_data, cat_indices_with_nulls) # Round values to match the original format
        self.is_fitted = True

        return imputed_data

    def transform(self, X, num_indices_with_nulls, cat_indices_with_nulls):
        """
        Impute missing values in the provided dataset using the trained model.

        Parameters:
        - X: numpy array of shape (n_samples, n_features)
            Dataset with missing values to impute.
        - num_indices_with_nulls: indices of numerical columns with nulls.
        - cat_indices_with_nulls: indices of categorical columns with nulls.

        Returns:
        - numpy array of shape (n_samples, n_features)
            Dataset with imputed values.
        """
        if not self.is_fitted:
            raise RuntimeError("The NOMIImputer must be fitted before calling transform.")

        data_x = X
        data_m = np.isnan(data_x)
        norm_data, norm_parameters = normalization(data_x)
        norm_data_x = np.nan_to_num(norm_data, 0)

        num, dims = norm_data_x.shape
        imputed_X = norm_data_x.copy()
        data_m_imputed = data_m.copy()
        col_indices_with_nulls = num_indices_with_nulls + cat_indices_with_nulls

        _, _, kernel_fn = stax.serial(
            stax.Dense(2 * dims), stax.Relu(),
            stax.Dense(dims), stax.Relu(),
            stax.Dense(1), stax.Sigmoid_like()
        )
        for dim in tqdm(col_indices_with_nulls, desc="Applying transform for each dimension"):
            X_wo_dim = np.delete(imputed_X, dim, 1)
            i_not_nan_index = data_m_imputed[:, dim].astype(bool)

            # Check for observed values
            if not np.any(i_not_nan_index):
                print(f"No observed values for dimension {dim}, skipping.")
                continue

            index = self.index_dct[dim]
            predict_fn = self.predict_fn_dct[dim]
            Y_train = self.Y_train_dct[dim]

            X_test = X_wo_dim[~i_not_nan_index]
            if X_test.shape[0] == 0:
                continue

            neigh_ind_test, neigh_dist_test = index.knn_query(X_test, k=self.k_neighbors, filter=None)
            neigh_dist_test = np.sqrt(neigh_dist_test)

            weights_test = dist2sim(neigh_dist_test[:, :-1])
            y_neighbors_test = Y_train[neigh_ind_test[:, :-1]]
            test_input = weights_test * y_neighbors_test

            y_pred, pred_cov = prediction(predict_fn, test_input, kernel_type="nngp")
            imputed_X[~i_not_nan_index, dim] = y_pred.reshape(-1)

        imputed_data = renormalization(imputed_X, norm_parameters)
        imputed_data = rounding(imputed_data, cat_indices_with_nulls)

        return imputed_data
