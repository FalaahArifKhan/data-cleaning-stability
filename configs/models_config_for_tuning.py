import numpy as np
from pprint import pprint
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier

from pytorch_tabular.models import GANDALFConfig
from pytorch_tabular.config import OptimizerConfig, TrainerConfig

from configs.constants import MLModels


def get_models_params_for_tuning(models_tuning_seed):
    return {
        MLModels.dt_clf.value: {
            'model': DecisionTreeClassifier(random_state=models_tuning_seed),
            'params': {
                "max_depth": [5, 10, 20, 30],
                'min_samples_leaf': [5, 10, 20, 50, 100],
                "max_features": [0.6, 'sqrt'],
                "criterion": ["gini", "entropy"]
            }
        },
        MLModels.lr_clf.value: {
            'model': LogisticRegression(random_state=models_tuning_seed, max_iter=1000),
            'params': {
                'penalty': ['l1', 'l2'],
                'C' : [0.001, 0.01, 0.1, 1],
                'solver': ['newton-cg', 'lbfgs', 'sag', 'saga'],
            }
        },
        MLModels.lgbm_clf.value: {
            'model': LGBMClassifier(random_state=models_tuning_seed, n_jobs=48, num_threads=48),
            'params': {
                'n_estimators': [50, 100, 200, 500],
                'max_depth' : [i for i in range(3, 10)] + [-1],
                'num_leaves' : [int(x) for x in np.linspace(start = 20, stop = 3000, num = 8)],
                'min_data_in_leaf' : [int(x) for x in np.linspace(start = 100, stop = 1000, num = 8)],
                'verbosity': [-1]
            }
        },
        MLModels.rf_clf.value: {
            'model': RandomForestClassifier(random_state=models_tuning_seed),
            'params': {
                'n_estimators': [50, 100, 200, 500, 700, 1000],
                # 'n_estimators': [50, 100, 200, 500],
                'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'bootstrap': [True, False]
            }
        },
        MLModels.mlp_clf.value: {
            'model': MLPClassifier(hidden_layer_sizes=(100,100,), random_state=models_tuning_seed, max_iter=1000),
            'params': {
                'activation': ['logistic', 'tanh', 'relu'],
                'solver': ['lbfgs', 'sgd', 'adam'],
                'learning_rate': ['constant', 'invscaling', 'adaptive']
            }
        },
        ####################################################################
        # Use Pytorch Tabular API to work with tabular neural networks
        ####################################################################
        MLModels.gandalf_clf.value: {
            'model': GANDALFConfig(task="classification", seed=models_tuning_seed),
            'optimizer_config': OptimizerConfig(),
            'trainer_config': TrainerConfig(batch_size=512,
                                            max_epochs=100,
                                            seed=models_tuning_seed,
                                            early_stopping="valid_loss", # Monitor valid_loss for early stopping
                                            early_stopping_mode="min", # Set the mode as min because for val_loss, lower is better
                                            early_stopping_patience=5), # No. of epochs of degradation training will wait before terminating
            'params': {
                'model_config__gflu_stages': [i for i in range(2, 31)],
                'model_config__gflu_dropout': [0.01 * i for i in range(6)],
                'model_config__gflu_feature_init_sparsity': [0.1 * i for i in range(6)],
                'model_config__learning_rate': [1e-3, 1e-4, 1e-5, 1e-6],
            }
        },
    }


if __name__ == '__main__':
    model_tuning_params = get_models_params_for_tuning(models_tuning_seed=200)
    pprint(model_tuning_params)
