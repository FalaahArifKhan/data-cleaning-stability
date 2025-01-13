"""
The original paper for the code below:
- Code: shared by authors over email

- Citation:
@article{miao2021efficient,
  title={Efficient and effective data imputation with influence functions},
  author={Miao, Xiaoye and Wu, Yangyang and Chen, Lu and Gao, Yunjun and Wang, Jun and Yin, Jianwei},
  journal={Proceedings of the VLDB Endowment},
  volume={15},
  number={3},
  pages={624--632},
  year={2021},
  publisher={VLDB Endowment}
}
"""
import math
import time
import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
from tqdm import tqdm
from itertools import product
from progress_bar import InitBar
from .nomi_imputer import rounding, normalization
from ..custom_classes.database_client import DatabaseClient
from configs.constants import (DIABETES_DATASET, GERMAN_CREDIT_DATASET, ACS_INCOME_DATASET, LAW_SCHOOL_DATASET,
                               BANK_MARKETING_DATASET, CARDIOVASCULAR_DISEASE_DATASET, ACS_EMPLOYMENT_DATASET)

tf.disable_eager_execution()


def tune_edit_gain(X, cat_indices_with_nulls: list, epochs: int, initial_sample_size: int,
                   validation_size: int, dataset_name: str, evaluation_scenario: str, seed: int):
    dataset_to_grid = {
        DIABETES_DATASET: {
            "gain": {
                "alpha": [1],
                "p_hint": [0.9],
            },
            "training": {
                "batch_size": [128],
            },
        },
        GERMAN_CREDIT_DATASET: {
            "gain": {
                "alpha": [1, 10],
                "p_hint": [0.7, 0.9],
            },
            "training": {
                "batch_size": [128],
            },
        },
        ACS_INCOME_DATASET: {
            "gain": {
                "alpha": [1, 10],
                "p_hint": [0.7, 0.9],
            },
            "training": {
                "batch_size": [128],
            },
        },
        LAW_SCHOOL_DATASET: {
            "gain": {
                "alpha": [1],
                "p_hint": [0.9],
            },
            "training": {
                "batch_size": [128],
            },
        },
        BANK_MARKETING_DATASET: {
            "gain": {
                "alpha": [1, 10],
                "p_hint": [0.7, 0.9],
            },
            "training": {
                "batch_size": [128],
            },
        },
        CARDIOVASCULAR_DISEASE_DATASET: {
            "gain": {
                "alpha": [1, 10],
                "p_hint": [0.7, 0.9],
            },
            "training": {
                "batch_size": [128],
            },
        },
        ACS_EMPLOYMENT_DATASET: {
            "gain": {
                "alpha": [1, 10],
                "p_hint": [0.7, 0.9],
            },
            "training": {
                "batch_size": [128],
            },
        },
    }

    db_client = DatabaseClient()
    db_client.connect()
    num_retries = 100 if dataset_name in (ACS_INCOME_DATASET, BANK_MARKETING_DATASET,
                                          CARDIOVASCULAR_DISEASE_DATASET, ACS_EMPLOYMENT_DATASET) else 30
    hyperparameter_grid = dataset_to_grid[dataset_name]
    all_combinations = create_all_combinations(hyperparameter_grid)
    for idx, combination in enumerate(all_combinations, 1):
        print(f"Combination {idx}: {combination}")
        imputer = EditGainImputer(batch_size=combination["batch_size"],
                                  alpha=combination["alpha"],
                                  epoch=epochs,
                                  p_hint=combination["p_hint"],
                                  initial_sample_size=initial_sample_size,
                                  validation_size=validation_size)

        for i in range(num_retries):
            try:
                start_training = time.time()
                imputer.fit(X)
                start_inference = time.time()
                X_np = imputer.transform(X, cat_indices_with_nulls)
                contains_nan = np.isnan(X_np).any()
                print("contains_nan:", contains_nan)
                if not contains_nan:
                    print(f"Final combination: {combination}")
                    imp_training_time = time.time() - start_training
                    imp_inference_time = time.time() - start_inference
                    print("imp_training_time:", imp_training_time)
                    print("imp_inference_time:", imp_inference_time)

                    imputation_metrics_df = pd.DataFrame({
                        "dataset_name": [dataset_name],
                        "null_imputer": ["edit_gain"],
                        "evaluation_scenario": [evaluation_scenario],
                        "experiment_seed": [seed],
                        "imp_training_time": [imp_training_time],
                        "imp_inference_time": [imp_inference_time],
                        "final_combination": [combination],
                    })
                    db_client.write_pandas_df_into_db(collection_name="edit_imp_performance_metrics",
                                                      df=imputation_metrics_df,
                                                      custom_tbl_fields_dct={})
                    db_client.close()
                    return imputer, X_np
            except Exception as e:
                print(f"Combination {idx}, retry {i + 1}: Failed with the following error {e}")

    return imputer, X_np


def renormalization(norm_data, norm_parameters):
    """Renormalize data from [0, 1] range to the original range.

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


def xavier_init(size, seed: int = 100):
    """Xavier initialization.

    Args:
      - size: vector size

    Returns:
      - initialized random vector.
    """
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    # return tf.random.stateless_normal(shape=size, stddev=xavier_stddev, seed=(seed, 0))
    return tf.random.normal(shape = size, stddev = xavier_stddev)


def binary_sampler(p, rows, cols):
    """Sample binary random variables.

    Args:
      - p: probability of 1
      - rows: the number of rows
      - cols: the number of columns

    Returns:
      - binary_random_matrix: generated binary random matrix.
    """
    np.random.seed(50)
    # np.random.seed(42)
    unif_random_matrix = np.random.uniform(0., 1., size=[rows, cols])
    # rng = np.random.default_rng(42)
    # unif_random_matrix = rng.uniform(0., 1., size=(rows, cols))
    binary_random_matrix = 1 * (unif_random_matrix < p)
    return binary_random_matrix


def sample_M(m, n, p):
    # Hint Vector Generation
    A = np.random.uniform(0., 1., size=[m, n])
    B = A > p
    C = 1. * B
    return C


def sample_Z(m, n, seed=None):
    # if seed is not None:
    #     rng = np.random.default_rng(seed)
    #     return rng.uniform(0., 0.01, size=(m, n))

    # Random sample generator for Z
    return np.random.uniform(0., 0.01, size=[m, n])


def sample_idx(m, n):
    # Mini-batch generation
    A = np.random.permutation(m)
    idx = A[:n]
    return idx


def sample_num_list(X_train, Initial_train_size: int, Validation_size: int):
    # np.random.seed(50)
    # return np.random.randint(len(X_train), size=Initial_train_size + Validation_size)
    rng = np.random.default_rng(42)
    return rng.integers(len(X_train), size=Initial_train_size + Validation_size)


def create_all_combinations(hyperparameter_grid):
    # Create all combinations
    keys, values = zip(*{
        key: list(product(*sub_dict.values())) if isinstance(sub_dict, dict) else sub_dict
        for key, sub_dict in hyperparameter_grid.items()
    }.items())

    nested_combinations = list(product(*values))

    # Flatten the nested combinations into a dictionary structure
    all_combinations = []
    for combo in nested_combinations:
        flat_combo = {}
        for i, (key, sub_values) in enumerate(zip(keys, combo)):
            if isinstance(hyperparameter_grid[key], dict):
                sub_keys = list(hyperparameter_grid[key].keys())
                flat_combo.update(dict(zip(sub_keys, sub_values)))
            else:
                flat_combo[key] = sub_values
        all_combinations.append(flat_combo)

    return all_combinations


class EditGainImputer:
    def __init__(self, batch_size: int, alpha: float, p_hint: float, epoch: int,
                 initial_sample_size: int, validation_size: int):
        self.mb_size = batch_size
        self.alpha = alpha
        self.p_hint = p_hint
        self.initial_sample_size = initial_sample_size
        self.validation_size = validation_size
        self.epoch = epoch

        self.gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = None
        self.norm_parameters = None
        self.is_fitted = False

    def _initial_model_train(self, sess, Dim, Initial_norm_data_x, Initial_data_m, M, X, New_X, H,
                             D_solver, D_loss1, G_solver, G_loss1, MSE_train_loss, MSE_test_loss):
        tf.disable_eager_execution()

        pbar = InitBar()
        num = 0
        for it in tqdm(range(self.epoch)):
            data_list = sample_idx(len(Initial_norm_data_x), len(Initial_norm_data_x))       # Mini batch
            mb_idx_list = []
            for i in range(0, len(data_list), self.mb_size):
                if i + self.mb_size > len(data_list):
                    break
                mb_idx_list.append(data_list[i:i + self.mb_size])
            for mb_idx in mb_idx_list:
                num += 1
                pbar(num / (self.epoch * len(mb_idx_list)) * 100)
                X_mb = Initial_norm_data_x[mb_idx, :]
                Z_mb = sample_Z(self.mb_size, Dim)
                M_mb = Initial_data_m[mb_idx, :]                            # data_m, Initial_data_m
                H_mb1 = sample_M(self.mb_size, Dim, 1 - self.p_hint)
                H_mb = M_mb * H_mb1
                New_X_mb = M_mb * X_mb + (1-M_mb) * Z_mb

                # Training discriminator
                _, D_loss_curr = sess.run([D_solver, D_loss1], feed_dict={M: M_mb, New_X: New_X_mb, H: H_mb})
                _, G_loss_curr, MSE_train_loss_curr, MSE_test_loss_curr = sess.run([G_solver, G_loss1, MSE_train_loss, MSE_test_loss],
                                                                                   feed_dict={X: X_mb, M: M_mb, New_X: New_X_mb, H: H_mb})
        return sess

    def _final_train_model(self, sess, Dim, final_x, final_m, X, New_X, M, H, D_solver, D_loss1,
                           G_solver, G_loss1, MSE_train_loss, MSE_test_loss):
        pbar = InitBar()
        num = 0
        for it in tqdm(range(self.epoch)):
            data_list = sample_idx(len(final_x), len(final_x))       # Mini batch
            mb_idx_list = []
            for i in range(0, len(data_list), self.mb_size):
                if i + self.mb_size > len(data_list):
                    break
                mb_idx_list.append(data_list[i:i + self.mb_size])
            for mb_idx in mb_idx_list:
                num += 1
                pbar(num / (self.epoch * len(mb_idx_list)) * 100)
                X_mb = final_x[mb_idx, :]
                Z_mb = sample_Z(self.mb_size, Dim)
                M_mb = final_m[mb_idx, :]                            # data_m, Initial_data_m
                H_mb1 = sample_M(self.mb_size, Dim, 1 - self.p_hint)
                H_mb = M_mb * H_mb1
                New_X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb
                # Training discriminator
                _, D_loss_curr = sess.run([D_solver, D_loss1], feed_dict={M: M_mb, New_X: New_X_mb, H: H_mb})
                _, G_loss_curr, MSE_train_loss_curr, MSE_test_loss_curr = sess.run([G_solver, G_loss1, MSE_train_loss, MSE_test_loss],
                                                                                   feed_dict={X: X_mb, M: M_mb, New_X: New_X_mb, H: H_mb})
        return sess

    def _calculate_each_sample_influence(self, sess, Dim, norm_data_x, data_m, Initial_norm_data_x, Initial_data_m,
                                         Val_norm_data_x, Val_data_m, H_invert, X, New_X, M, H,
                                         final_zhuan, what, Len_matrix, final):
        # Inverted Hessian Matrix Calculation
        Z_mb = sample_Z(len(Initial_norm_data_x), Dim)
        M_mb = Initial_data_m
        X_mb = Initial_norm_data_x
        New_X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb  # Missing Data Introduce
        H_mb_temp = binary_sampler(self.p_hint, len(Initial_data_m), Dim)
        H_mb = M_mb * H_mb_temp

        true_inv_h = sess.run([H_invert], feed_dict={X: X_mb, M: M_mb, New_X: New_X_mb, H: H_mb})  # The hessian of Initial set

        # Calculate the gradient of validation set
        Z_mb = sample_Z(len(Val_norm_data_x), Dim)
        M_mb = Val_data_m
        X_mb = Val_norm_data_x
        New_X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb  # Missing Data Introduce
        H_mb_temp = binary_sampler(self.p_hint, len(Val_data_m), Dim)
        H_mb = M_mb * H_mb_temp
        Val_grad = sess.run([final_zhuan], feed_dict={X: X_mb, M: M_mb, New_X: New_X_mb, H: H_mb})  # The hessian of Validation set
        Val_loss = sess.run([what], feed_dict={X: X_mb, M: M_mb, New_X: New_X_mb, H: H_mb})  # The hessian of Validation set
        # print("Val_grad:\n", Val_grad[0])
        IHVP = [np.dot(Val_grad[0][i], true_inv_h[0][i]) for i in range(Len_matrix)]

        print('**************************************')

        # Calculate the influence of each training point
        val_lissa = []
        value = []
        pbar = InitBar()
        number = len(data_m)
        for i in range(number):
            pbar(i / number * 100)
            M_mb = np.array([data_m[i]])           # data_m, Initial_data_m
            X_mb = np.array([norm_data_x[i]])      # norm_data_x, Initial_norm_data_x
            New_X_mb = M_mb * X_mb + (1 - M_mb) * sample_Z(1, Dim)  # Missing Data Introduce
            H_mb = M_mb * binary_sampler(self.p_hint, 1, Dim)
            train_grad = sess.run([final], feed_dict={X: X_mb, M: M_mb, New_X: New_X_mb, H: H_mb})
            if math.isnan(np.dot(np.concatenate(IHVP, axis=1), np.concatenate(train_grad[0], axis=0))[0][0]):
                val_lissa.append([i, 0])
                value.append(0)
            else:
                val_lissa.append([i, np.dot(np.concatenate(IHVP, axis=1), np.concatenate(train_grad[0], axis=0))[0][0]])
                value.append(np.dot(np.concatenate(IHVP, axis=1), np.concatenate(train_grad[0], axis=0))[0][0])

        # print('\n Multiplying by %s train examples took %s minute %s sec' % (1, duration // 60, duration % 60))
        val_lissa = sorted(val_lissa, key=lambda x: x[1], reverse=True)
        influence_sum = sum(value)
        # print('influence_sum:', influence_sum)
        influence_top = 0
        Top_k_list = []
        for i in range(len(norm_data_x)):
            if influence_top > influence_sum:
                break
            else:
                influence_top += float(val_lissa[i][1])
                #print('influence_top: ', influence_top)
                Top_k_list.append(val_lissa[i][0])

        print('**************************************', len(Top_k_list))
        return Top_k_list, sess

    def _train_edit_gain_model(self, Dim, norm_data_x, data_m, Initial_norm_data_x, Initial_data_m,
                               Val_norm_data_x, Val_data_m):
        damping = 1e-2
        number_list = []
        H_Dim1 = Dim
        H_Dim2 = Dim

        ### 1. Input Placeholders
        # 1.1. Data Vector
        X = tf.placeholder(tf.float32, shape=[None, Dim])
        # 1.2. Mask Vector
        M = tf.placeholder(tf.float32, shape=[None, Dim])
        # 1.3. Hint vector
        H = tf.placeholder(tf.float32, shape=[None, Dim])
        # 1.4. X with missing values
        New_X = tf.placeholder(tf.float32, shape=[None, Dim])
    
        ### 2. Discriminator
        D_W1 = tf.Variable(xavier_init([Dim*2, H_Dim1], seed=123))     # Data + Hint as inputs
        D_b1 = tf.Variable(tf.zeros(shape=[H_Dim1]))
    
        D_W2 = tf.Variable(xavier_init([H_Dim1, H_Dim2], seed=234))
        D_b2 = tf.Variable(tf.zeros(shape=[H_Dim2]))
    
        D_W3 = tf.Variable(xavier_init([H_Dim2, Dim], seed=345))
        D_b3 = tf.Variable(tf.zeros(shape=[Dim]))       # Output is multi-variate
    
        theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3]
    
        ### 3. Generator
        G_W1 = tf.Variable(xavier_init([Dim*2, H_Dim1], seed=456))     # Data + Mask as inputs (Random Noises are in Missing Components)
        G_b1 = tf.Variable(tf.zeros(shape=[H_Dim1]))
    
        G_W2 = tf.Variable(xavier_init([H_Dim1, H_Dim2], seed=567))
        G_b2 = tf.Variable(tf.zeros(shape=[H_Dim2]))
    
        G_W3 = tf.Variable(xavier_init([H_Dim2, Dim], seed=678))
        G_b3 = tf.Variable(tf.zeros(shape=[Dim]))

        theta_G = [G_W1, G_W2, G_W3, G_b1, G_b2, G_b3]
    
        # GAIN Function
    
        # 1. Generator
        def generator(new_x, m):
            inputs = tf.concat(axis=1, values=[new_x, m])  # Mask + Data Concatenate
            G_h1 = tf.nn.relu(tf.matmul(inputs, G_W1) + G_b1)
            G_h2 = tf.nn.relu(tf.matmul(G_h1, G_W2) + G_b2)
            G_prob = tf.nn.sigmoid(tf.matmul(G_h2, G_W3) + G_b3)  # [0,1] normalized Output
    
            return G_prob
    
        # 2. Discriminator
        def discriminator(new_x, h):
            inputs = tf.concat(axis=1, values=[new_x, h])  # Hint + Data Concatenate
            D_h1 = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)
            D_h2 = tf.nn.relu(tf.matmul(D_h1, D_W2) + D_b2)
            D_logit = tf.matmul(D_h2, D_W3) + D_b3
            D_prob = tf.nn.sigmoid(D_logit)  # [0,1] Probability Output
    
            return D_prob
    
        ### Structure
        # Generator
        G_sample = generator(New_X, M)                # Definition of generator: input X without missing values and Mask Vector
    
        # Combine with original data
        Hat_New_X = New_X * M + G_sample * (1 - M)     # Generated complete data
    
        # Discriminator
        D_prob = discriminator(Hat_New_X, H)         # Definition of generator: input generated data and Hint Vector
    
        ### Loss
        D_loss1 = -tf.reduce_mean(M * tf.log(D_prob + 1e-8) + (1-M) * tf.log(1. - D_prob + 1e-8))
        G_loss1 = -tf.reduce_mean((1-M) * tf.log(D_prob + 1e-8))
        MSE_train_loss = tf.reduce_mean((M * New_X - M * G_sample)**2) / tf.reduce_mean(M)
    
        D_loss = D_loss1
        G_loss = G_loss1 + self.alpha * MSE_train_loss + damping * tf.reduce_mean([tf.nn.l2_loss(i) for i in theta_G])
    
        ### MSE Performance metric
        MSE_test_loss = tf.reduce_mean(((1-M) * X - (1-M)*G_sample)**2) / tf.reduce_mean(1-M)
        MAE_test_loss = tf.reduce_mean((((1-M) * X - (1-M)*G_sample)**2)**0.5) / tf.reduce_mean(1-M)
        what = tf.reduce_mean(1-M)
    
        ####### Fast Hessians Calculation
        H_gradient_ = tf.gradients(G_loss, theta_G)
        # Hessians = tf.hessians(G_loss, theta_G)
        Len_matrix = len(theta_G)
        final = [tf.reshape(H_gradient_[i], [-1, 1]) for i in range(Len_matrix)]
        final_zhuan = [tf.transpose(tf.reshape(H_gradient_[i], [-1, 1])) for i in range(Len_matrix)]
        Fast_Hessians = [final[i] * final_zhuan[i] for i in range(Len_matrix)]
        H_hessians = [Fast_Hessians[num] + tf.eye(tf.shape(Fast_Hessians[num])[0]) * 10e-4 for num in range(Len_matrix)]
        H_invert = [tf.linalg.inv(item + 10e-4) for item in H_hessians]
    
        ### Solver
        D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)   # Optimizer for D
        G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)   # Optimizer for G

        # Sessions Definition
        sess = tf.Session(config=tf.ConfigProto(gpu_options=self.gpu_options))
        sess.run(tf.global_variables_initializer())

        # ========================================================================
        # Initial Model Training
        # ========================================================================
        sess = self._initial_model_train(sess=sess,
                                         Dim=Dim,
                                         Initial_norm_data_x=Initial_norm_data_x,
                                         Initial_data_m=Initial_data_m,
                                         M=M, X=X, New_X=New_X, H=H,
                                         D_solver=D_solver,
                                         D_loss1=D_loss1,
                                         G_solver=G_solver,
                                         G_loss1=G_loss1,
                                         MSE_train_loss=MSE_train_loss,
                                         MSE_test_loss=MSE_test_loss)

        # ========================================================================
        # Imputation Influence Function
        # ========================================================================
        s = time.time()
        # num_retries = 100
        # for i in range(num_retries):
        #     try:
        Top_k_list, sess = self._calculate_each_sample_influence(sess=sess,
                                                                 Dim=Dim,
                                                                 norm_data_x=norm_data_x,
                                                                 data_m=data_m,
                                                                 Initial_norm_data_x=Initial_norm_data_x,
                                                                 Initial_data_m=Initial_data_m,
                                                                 Val_norm_data_x=Val_norm_data_x,
                                                                 Val_data_m=Val_data_m,
                                                                 H_invert=H_invert,
                                                                 X=X, New_X=New_X, M=M, H=H,
                                                                 final_zhuan=final_zhuan,
                                                                 what=what,
                                                                 Len_matrix=Len_matrix,
                                                                 final=final)
            #     break
            # except Exception as e:
            #     print(f"Retry {i + 1} failed")

        number_list.append(len(Top_k_list))
        duration = time.time() - s
        print('Time: ', duration)

        # ========================================================================
        # Final Train Model
        # ========================================================================
        final_x = norm_data_x[Top_k_list]
        final_m = data_m[Top_k_list]
        sess = self._final_train_model(sess=sess,
                                       Dim=Dim,
                                       final_x=final_x,
                                       final_m=final_m,
                                       X=X, New_X=New_X, M=M, H=H,
                                       D_solver=D_solver,
                                       D_loss1=D_loss1,
                                       G_solver=G_solver,
                                       G_loss1=G_loss1,
                                       MSE_train_loss=MSE_train_loss,
                                       MSE_test_loss=MSE_test_loss)

        return sess, X, New_X, M, H, G_sample, G_loss1

    def fit(self, X_train):
        """
        Fit the EDIT GAIN imputer using the provided training data.
        """
        # Configs
        Initial_train_size = min(self.initial_sample_size, X_train.shape[0] // 2)
        Validation_size = min(self.validation_size, X_train.shape[0] // 2)
        num_list = np.random.randint(len(X_train), size=Initial_train_size + Validation_size)
        # num_list = sample_num_list(X_train, Initial_train_size, Validation_size)

        Initial_list = num_list[:Initial_train_size]
        Validation_list = num_list[Initial_train_size:]
        Dim = len(X_train[0, :])

        # Data preparation
        Initial_data = X_train[Initial_list]
        Initial_data_m = 1 - np.isnan(Initial_data)

        Val_data = X_train[Validation_list]
        Val_data_m = 1 - np.isnan(Val_data)

        data_x = X_train
        data_m = 1 - np.isnan(data_x)

        norm_data, self.norm_parameters = normalization(data_x)
        norm_data_x = np.nan_to_num(norm_data, 0)

        Initial_norm_data, Initial_norm_parameters = normalization(Initial_data)
        Initial_norm_data_x = np.nan_to_num(Initial_norm_data, 0)

        Val_norm_data, Val_norm_parameters = normalization(Val_data)
        Val_norm_data_x = np.nan_to_num(Val_norm_data, 0)

        # Train the imputer using EDIT
        sess, X, New_X, M, H, G_sample, G_loss1 = self._train_edit_gain_model(Dim=Dim,
                                                                              norm_data_x=norm_data_x,
                                                                              data_m=data_m,
                                                                              Initial_norm_data_x=Initial_norm_data_x,
                                                                              Initial_data_m=Initial_data_m,
                                                                              Val_norm_data_x=Val_norm_data_x,
                                                                              Val_data_m=Val_data_m)
        self.sess = sess
        self.X = X
        self.New_X = New_X
        self.M = M
        self.H = H
        self.G_sample = G_sample
        self.G_loss1 = G_loss1

        self.is_fitted = True
        return self

    def transform(self, X_test, cat_indices_with_nulls):
        """
        Impute missing values in the provided dataset using the trained model.
        """
        if not self.is_fitted:
            raise RuntimeError("The NOMIImputer must be fitted before calling transform.")

        Dim = len(X_test[0, :])
        data_x = X_test
        data_m = 1 - np.isnan(data_x)
        norm_data, _ = normalization(data_x, self.norm_parameters)
        norm_data_x = np.nan_to_num(norm_data, 0)

        # Z_mb = sample_Z(len(data_x), Dim, seed=100)
        Z_mb = sample_Z(len(data_x), Dim)
        M_mb = data_m
        X_mb = norm_data_x
        New_X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb  # Missing Data Introduce

        # MSE_final is the output of GAIN,and Sample is the output of generator
        Sample = self.sess.run([self.G_sample], feed_dict={self.X: norm_data_x, self.M: data_m, self.New_X: New_X_mb})

        # Perform imputation
        imputed_data = data_m * norm_data_x + (1 - data_m) * Sample[0]

        imputed_data = renormalization(imputed_data, self.norm_parameters) # Re-normalize the imputed data
        imputed_data = rounding(imputed_data, cat_indices_with_nulls) # Rounding

        return imputed_data
