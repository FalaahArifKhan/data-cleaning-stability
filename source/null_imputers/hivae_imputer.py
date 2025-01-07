import os

import tensorflow as tf
import numpy as np

import external_dependencies.HIVAE.graph_new as graph_new # Importing external module for graph creation


class HIVAEImputer:
    def __init__(self, dim_latent_z=2, dim_latent_y=3, dim_latent_s=4, batch_size=128, epochs=100, learning_rate=1e-3):
        self.dim_latent_z = dim_latent_z
        self.dim_latent_y = dim_latent_y
        self.dim_latent_s = dim_latent_s
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.trained = False
        
        # Placeholder for the model
        self.model = None

    def build_model(self, types_file):
        # Build TensorFlow computational graph
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.tf_nodes = self._build_hvae_graph(types_file)

    def _build_hvae_graph(self, types_file):
        """
        Build the computational graph for the HI-VAE model using the provided types file.

        Args:
            types_file: str, path to the file describing data types.

        Returns:
            dict: Dictionary containing TensorFlow nodes for input, output, loss, etc.
        """
        graph_nodes = graph_new.HVAE_graph(
            model_name='HIVAE',
            types_file=types_file,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            z_dim=self.dim_latent_z,
            y_dim=self.dim_latent_y,
            s_dim=self.dim_latent_s
        )
        return graph_nodes

    def fit(self, X_train, mask_train):
        """
        Train the HIVAE model on the provided training data.

        Args:
            X_train: np.array, the input data with missing values.
            mask_train: np.array, a mask array where 1 indicates observed values and 0 indicates missing values.
        """
        if not self.trained:
            self._initialize_training(X_train, mask_train)

        with tf.Session(graph=self.graph) as session:
            tf.global_variables_initializer().run()
            saver = tf.train.Saver()

            for epoch in range(self.epochs):
                avg_loss = 0.0

                for batch_data, batch_mask in self._batch_iterator(X_train, mask_train):
                    feed_dict = {
                        self.tf_nodes['input']: batch_data,
                        self.tf_nodes['mask']: batch_mask
                    }
                    _, loss = session.run([self.tf_nodes['optimizer'], self.tf_nodes['loss']], feed_dict=feed_dict)
                    avg_loss += loss

                avg_loss /= len(X_train) // self.batch_size
                print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {avg_loss}")

            saver.save(session, './hvae_model.ckpt')
            self.trained = True

    def transform(self, X_test, mask_test):
        """
        Perform imputation on the test data using the trained model.

        Args:
            X_test: np.array, the test data with missing values.
            mask_test: np.array, a mask array where 1 indicates observed values and 0 indicates missing values.

        Returns:
            np.array: The imputed test data.
        """
        if not self.trained:
            raise ValueError("Model must be trained with `fit` before calling `transform`.")

        with tf.Session(graph=self.graph) as session:
            saver = tf.train.Saver()
            saver.restore(session, './hvae_model.ckpt')

            imputed_data = []

            for batch_data, batch_mask in self._batch_iterator(X_test, mask_test):
                feed_dict = {
                    self.tf_nodes['input']: batch_data,
                    self.tf_nodes['mask']: batch_mask
                }
                imputed_batch = session.run(self.tf_nodes['imputed_output'], feed_dict=feed_dict)
                imputed_data.append(imputed_batch)

            return np.vstack(imputed_data)

    def _batch_iterator(self, data, mask):
        """
        A generator that yields batches of data and their corresponding masks.

        Args:
            data: np.array, the dataset.
            mask: np.array, the mask array.

        Yields:
            tuple: A batch of data and its mask.
        """
        n_samples = data.shape[0]
        indices = np.arange(n_samples)
        np.random.shuffle(indices)

        for start_idx in range(0, n_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]
            yield data[batch_indices], mask[batch_indices]

    def _initialize_training(self, X_train, mask_train):
        """
        Initialize model parameters and data preprocessing steps.

        Args:
            X_train: np.array, training data.
            mask_train: np.array, training mask.
        """
        print("Initializing training setup...")
        self.input_dim = X_train.shape[1]
        self.n_samples = X_train.shape[0]

        # Initialize graph and nodes if not already done
        if not self.graph:
            raise RuntimeError("Graph must be built using `build_model` before initializing training.")

        print(f"Training setup: {self.n_samples} samples, {self.input_dim} dimensions.")
