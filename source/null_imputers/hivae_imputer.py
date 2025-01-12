import numpy as np
import tensorflow as tf
import time
import os

import external_dependencies.HIVAE.graph_new as graph_new


class HIVAEImputer:
    """
    A class-based interface for training and testing the HI-VAE model
    using graph_new.HVAE_graph in TF1.x style, without reading data from files.

    Usage example:
        # Suppose you have:
        #   X_train: np.array of shape [N_train, D]
        #   mask_train: np.array of shape [N_train, D] (1=observed, 0=missing)
        #   types_dict: list of dictionaries describing each column
        #       e.g., types_dict[0] = {'type': 'cat', 'dim': 3}
        #
        imputer = HIVAEImputer(
            dim_latent_z=2,
            dim_latent_y=3,
            dim_latent_s=4,
            batch_size=128,
            epochs=100,
            learning_rate=1e-3,
            model_name='model_HIVAE_inputDropout',
            checkpoint_path='./hvae_model.ckpt'
        )
        imputer.build_model(types_file='path/to/data_types.csv')  # or another CSV
        imputer.fit(X_train, mask_train, types_dict)
        X_imputed = imputer.transform(X_test, mask_test, types_dict)

    Note on 'types_file': The original HVAE code typically loads variable info
    (like #dimensions) from CSV. You may pass any valid 'types_file' path if 
    your 'graph_new.HVAE_graph' requires it. The 'types_dict' given to fit/transform
    must match that same structure (same column ordering, 'type', 'dim', etc.).
    """

    def __init__(
        self,
        dim_latent_z=2,
        dim_latent_y=3,
        dim_latent_s=4,
        batch_size=128,
        epochs=100,
        learning_rate=1e-3,
        model_name='model_HIVAE_inputDropout',
        checkpoint_path='./hvae_model.ckpt'
    ):
        self.dim_latent_z = dim_latent_z
        self.dim_latent_y = dim_latent_y
        self.dim_latent_s = dim_latent_s
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.model_name = model_name

        self.checkpoint_path = checkpoint_path
        self.trained = False
        self.graph = None
        self.tf_nodes = {}

    # -------------------------------------------------------------------------
    #                           Data Encoding / Decoding
    # -------------------------------------------------------------------------
    def _encode_data(self, X, types_dict):
        """
        Encodes raw data X (shape [N, D]) into expanded form (shape [N, sum_of_dims])
        depending on each variable's type in types_dict.

        For example:
          - 'cat': one-hot
          - 'ordinal': "thermometer" encoding
          - 'count': direct pass, but if min=0 => we might do +1 if needed
          - 'pos' or 'real': direct pass

        Returns:
            X_enc: np.array, shape [N, total_expanded_dim]
        """
        N, D = X.shape
        if D != len(types_dict):
            raise ValueError(f"X has {D} columns but types_dict has {len(types_dict)} entries.")

        encoded_cols = []

        for col_idx, col_info in enumerate(types_dict):
            col_type = col_info['type']
            col_dim = int(col_info['dim'])

            # Extract this column from X
            col_data = X[:, col_idx]

            if col_type == 'cat':
                # Convert to int, then one-hot
                # If col_data has categories like {0,1,2} we do one-hot of length col_dim
                # (Assumes col_dim matches #unique categories).
                col_data_int = col_data.astype(int)
                # Guard: if values exceed col_dim-1, or negative, fix or raise error
                one_hot = np.zeros((N, col_dim))
                for i in range(N):
                    cat_val = col_data_int[i]
                    if 0 <= cat_val < col_dim:
                        one_hot[i, cat_val] = 1.0
                    else:
                        # If there's out-of-bound category => clamp or raise an error
                        pass
                encoded_cols.append(one_hot)

            elif col_type == 'ordinal':
                # "Thermometer" encoding for ordinal in {0,1,...,col_dim-1}
                # If col_dim=5, then possible categories are [0..4].
                # We expand to shape (N, col_dim). Each row i has 1's up to the category index, else 0.
                # But the original HVAE code uses a cumsum approach with an extra column.
                # We'll replicate the same logic used in read_functions:
                #    we create array (N, 1 + col_dim), fill the first col with 1,
                #    fill index (1 + cat_val) with -1, then cumsum along axis=1, drop the last col.
                col_data_int = col_data.astype(int)
                # Construct an array of shape (N, 1 + col_dim)
                arr = np.zeros((N, 1 + col_dim))
                arr[:, 0] = 1.0  # first col = 1
                for i in range(N):
                    cat_val = col_data_int[i]
                    # clamp cat_val to [0..col_dim-1] if needed
                    if cat_val < 0: 
                        cat_val = 0
                    elif cat_val >= col_dim:
                        cat_val = col_dim - 1
                    arr[i, 1 + cat_val] = -1.0
                arr = np.cumsum(arr, axis=1)
                # Drop last column => shape (N, col_dim)
                thermometer = arr[:, :-1]
                encoded_cols.append(thermometer)

            elif col_type == 'count':
                # read_functions suggests if min(data[:,i]) == 0 => add +1
                # We'll replicate that logic if needed. 
                if np.min(col_data) == 0:
                    col_data_adj = col_data + 1.0
                else:
                    col_data_adj = col_data
                # shape (N,1)
                encoded_cols.append(col_data_adj.reshape(-1, 1))

            else:
                # 'real', 'pos', or anything else => direct pass as (N,1) or (N,col_dim)
                # E.g., 'pos' might do log transform, but in original code we just store as (N,1).
                encoded_cols.append(col_data.reshape(-1, 1))

        # Concatenate all expansions
        X_enc = np.concatenate(encoded_cols, axis=1)
        return X_enc

    def _decode_data(self, X_enc, types_dict):
        """
        Inverse of _encode_data. For cat => argmax. For ordinal => sum(...) - 1. 
        For 'real', 'pos', 'count' => pass through.

        X_enc is shape [N, sum_of_dims]. We split it according to types_dict, then decode.
        Returns a shape [N, D] array in the *original* column arrangement.

        This replicates discrete_variables_transformation(...) from read_functions.
        """
        N = X_enc.shape[0]
        idx_start = 0
        out_cols = []

        for col_info in types_dict:
            col_type = col_info['type']
            col_dim = int(col_info['dim'])

            if col_type == 'cat':
                # slice out col_dim columns, do argmax => shape [N,1]
                slice_ = X_enc[:, idx_start : idx_start + col_dim]
                col_decoded = np.argmax(slice_, axis=1).reshape(-1, 1)
                idx_start += col_dim

            elif col_type == 'ordinal':
                # slice out col_dim columns, sum(...) => range 0..(col_dim-1).
                slice_ = X_enc[:, idx_start : idx_start + col_dim]
                # original code: output = sum(slice_, axis=1) - 1
                # but we must ensure it doesn't go below 0
                col_decoded = (np.sum(slice_, axis=1) - 1).reshape(-1, 1)
                idx_start += col_dim

            else:
                # 'count', 'real', 'pos', etc. => just pass the slice
                slice_ = X_enc[:, idx_start : idx_start + col_dim]
                col_decoded = slice_
                idx_start += col_dim

            out_cols.append(col_decoded)

        X_dec = np.concatenate(out_cols, axis=1)
        return X_dec

    def _split_data_by_variable(self, X_enc, types_dict):
        """
        Splits the encoded data X_enc (shape [N, sum_of_dims]) into a list of length len(types_dict),
        where each entry has shape [N, var_dim].

        The original HVAE code expects something like data_list[i] = X_enc[:, some_range].
        """
        data_list = []
        idx_start = 0

        for col_info in types_dict:
            var_dim = int(col_info['dim'])
            # For 'cat' => var_dim is the one-hot dimension, for 'ordinal' => var_dim is the thermometer dimension, etc.
            # But notice 'ordinal' is coded with col_dim columns (like a cumsum minus 1). 
            #   so var_dim in types_dict must match the expansions we did in _encode_data.
            #   (the read_functions code uses the same approach.)
            slice_ = X_enc[:, idx_start : idx_start + var_dim]
            data_list.append(slice_)
            idx_start += var_dim

        return data_list

    def _batch_iterator(self, X_enc, mask, types_dict):
        """
        A generator that yields (data_list, miss_list) for each mini-batch.

        Here:
         - X_enc: encoded data, shape [N, sum_of_dims].
         - mask: shape [N, D], 1=observed, 0=missing.
         - types_dict: used to split X_enc into a list of sub-arrays.

        We do random shuffling per epoch in fit(...).
        """
        N = X_enc.shape[0]
        n_batches = (N + self.batch_size - 1) // self.batch_size

        for i in range(n_batches):
            start = i * self.batch_size
            end = min(start + self.batch_size, N)

            batch_X_enc = X_enc[start:end]
            batch_mask = mask[start:end]

            # Split the encoded data into a list
            batch_data_list = self._split_data_by_variable(batch_X_enc, types_dict)

            # miss_list is shape [batch_size, D], consistent with HVAE placeholders
            yield batch_data_list, batch_mask

    # -------------------------------------------------------------------------
    #                       Graph Construction & Training
    # -------------------------------------------------------------------------
    def build_model(self, types_file, y_dim_partition=None):
        """
        Build the TensorFlow computational graph by calling graph_new.HVAE_graph.
        'types_file' is a CSV describing variable types. Must be consistent with 'types_dict'.
        """
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.tf_nodes = graph_new.HVAE_graph(
                model_name=self.model_name,
                types_file=types_file,
                batch_size=self.batch_size,
                learning_rate=self.learning_rate,
                z_dim=self.dim_latent_z,
                y_dim=self.dim_latent_y,
                s_dim=self.dim_latent_s,
                y_dim_partition=y_dim_partition  # if needed
            )

    def fit(self, X, mask, types_dict, display_epoch=1):
        """
        Train HI-VAE on X (shape [N, D]) with a given mask (shape [N, D]) and a types_dict
        describing each column. We encode X, then feed mini-batches to HVAE placeholders.

        Args:
            X: (N, D) raw data array (may have real/cat/ordinal/etc.)
            mask: (N, D), 1=observed, 0=missing
            types_dict: list of length D, each a dict like {'type': 'cat', 'dim': 3}, etc.
            display_epoch: int, how often to print training info
        """
        if self.graph is None:
            raise RuntimeError(
                "Graph not built. Call `build_model(...)` before calling `fit`."
            )

        # Encode data to the expanded representation
        X_enc = self._encode_data(X, types_dict)
        N = X_enc.shape[0]
        n_batches = (N + self.batch_size - 1) // self.batch_size

        saver = None
        with tf.Session(graph=self.graph) as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()

            print("Start HI-VAE training ...")
            start_time = time.time()

            for epoch in range(self.epochs):
                # Shuffle data each epoch
                indices = np.random.permutation(N)
                X_enc_epoch = X_enc[indices]
                mask_epoch = mask[indices]

                # Gumbel-Softmax annealing if needed
                tau = max(1.0 - 0.01 * epoch, 1e-3)  # example schedule
                tau2 = min(0.001 * epoch, 1.0)

                avg_loss = 0.0
                avg_kl_s = 0.0
                avg_kl_z = 0.0

                # Mini-batch training
                batch_count = 0
                for batch_data_list, batch_mask in self._batch_iterator(X_enc_epoch, mask_epoch, types_dict):
                    feed_dict = {}
                    # 'ground_batch' placeholders
                    for j, ph in enumerate(self.tf_nodes['ground_batch']):
                        feed_dict[ph] = batch_data_list[j]
                    # 'ground_batch_observed' placeholders: multiply data by mask
                    for j, ph in enumerate(self.tf_nodes['ground_batch_observed']):
                        feed_dict[ph] = batch_data_list[j] * np.reshape(batch_mask[:, j], [-1, 1])

                    feed_dict[self.tf_nodes['miss_list']] = batch_mask
                    feed_dict[self.tf_nodes['tau_GS']] = tau
                    if 'tau_var' in self.tf_nodes:  # if present
                        feed_dict[self.tf_nodes['tau_var']] = tau2

                    # Run optimizer and get losses
                    _, loss_re, kl_z, kl_s = sess.run(
                        [
                            self.tf_nodes['optim'],
                            self.tf_nodes['loss_re'],
                            self.tf_nodes['KL_z'],
                            self.tf_nodes['KL_s']
                        ],
                        feed_dict=feed_dict
                    )

                    avg_loss += loss_re
                    avg_kl_z += kl_z
                    avg_kl_s += kl_s
                    batch_count += 1

                # Average stats
                avg_loss /= batch_count
                avg_kl_z /= batch_count
                avg_kl_s /= batch_count

                if (epoch + 1) % display_epoch == 0:
                    elapsed = time.time() - start_time
                    elbo = avg_loss - avg_kl_z - avg_kl_s
                    print(
                        f"Epoch: [{epoch+1}/{self.epochs}]  "
                        f"time: {elapsed:.1f}, "
                        f"train_loglik: {avg_loss:.6f}, "
                        f"KL_z: {avg_kl_z:.6f}, "
                        f"KL_s: {avg_kl_s:.6f}, "
                        f"ELBO: {elbo:.6f}"
                    )

            # Save model
            saver.save(sess, self.checkpoint_path)
            self.trained = True
            print("Training finished. Model saved at:", self.checkpoint_path)

    def transform(self, X, mask, types_dict):
        """
        Impute/transform on test data. Restores the trained model, encodes X, 
        runs 'samples_test' from the graph, then decodes it back to original space.

        Args:
            X: shape [N, D], raw data
            mask: shape [N, D], 1=observed, 0=missing
            types_dict: same dictionary used in fit()

        Returns:
            X_imputed: shape [N, D], the imputed data in original dimension
        """
        if not self.trained:
            raise ValueError("Model must be trained (`fit`) before calling `transform`.")
        if self.graph is None:
            raise RuntimeError("Graph not built. Call `build_model(...)` first.")

        X_enc = self._encode_data(X, types_dict)
        N = X_enc.shape[0]
        n_batches = (N + self.batch_size - 1) // self.batch_size

        imputed_enc_list = []
        saver = None

        with tf.Session(graph=self.graph) as sess:
            saver = tf.train.Saver()
            saver.restore(sess, self.checkpoint_path)
            print("Model restored for inference from", self.checkpoint_path)

            # For inference, we often fix tau=1e-3
            tau = 1e-3

            for batch_data_list, batch_mask in self._batch_iterator(X_enc, mask, types_dict):
                feed_dict = {}
                # 'ground_batch' placeholders
                for j, ph in enumerate(self.tf_nodes['ground_batch']):
                    feed_dict[ph] = batch_data_list[j]
                # 'ground_batch_observed'
                for j, ph in enumerate(self.tf_nodes['ground_batch_observed']):
                    feed_dict[ph] = batch_data_list[j] * np.reshape(batch_mask[:, j], [-1, 1])

                feed_dict[self.tf_nodes['miss_list']] = batch_mask
                feed_dict[self.tf_nodes['tau_GS']] = tau

                # Run the "samples_test" node to get the imputed data in encoded space
                batch_imputed_enc = sess.run(
                    self.tf_nodes['samples_test'],
                    feed_dict=feed_dict
                )
                imputed_enc_list.append(batch_imputed_enc)

        # Concatenate batch results
        imputed_enc = np.concatenate(imputed_enc_list, axis=0)
        # Trim to original N if there's any leftover
        imputed_enc = imputed_enc[:N]

        # Decode back to original dimension
        X_imputed = self._decode_data(imputed_enc, types_dict)
        return X_imputed

    # -------------------------------------------------------------------------
    #                  Optional: Basic Mean Imputation Baseline
    # -------------------------------------------------------------------------
    def mean_imputation(self, X, mask, types_dict):
        """
        Simple baseline: fill missing entries using the mean (numeric) or mode (cat/ordinal).
        This replicates 'mean_imputation(...)' from read_functions, 
        but directly on your arrays (X, mask).
        """
        N, D = X.shape
        X_filled = X.copy()

        for d, col_info in enumerate(types_dict):
            col_type = col_info['type']
            col_dim = int(col_info['dim'])
            # Indices where the column is observed vs missing
            obs_inds = np.where(mask[:, d] == 1)[0]
            miss_inds = np.where(mask[:, d] == 0)[0]

            if len(miss_inds) == 0:
                continue  # No missing in this column, skip

            if col_type in ['cat', 'ordinal']:
                # Mode of observed
                obs_vals = X_filled[obs_inds, d]
                # convert to int
                obs_vals = obs_vals.astype(int)
                vals, counts = np.unique(obs_vals, return_counts=True)
                mode_val = vals[np.argmax(counts)]
                # fill
                X_filled[miss_inds, d] = mode_val
            else:
                # numeric => mean
                obs_vals = X_filled[obs_inds, d]
                mean_val = np.mean(obs_vals)
                X_filled[miss_inds, d] = mean_val

        return X_filled
