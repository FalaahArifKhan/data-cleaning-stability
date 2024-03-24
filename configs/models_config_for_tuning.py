import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier


def get_models_params_for_tuning(models_tuning_seed):
    return {
        'LGBMClassifier': {
            'model': LGBMClassifier(random_state=models_tuning_seed),
            'params': {
                'max_depth' : [i for i in range(3,12)],
                'num_leaves' : [int(x) for x in np.linspace(start = 20, stop = 3000, num = 10)],
                'min_data_in_leaf' : [int(x) for x in np.linspace(start = 100, stop = 1000, num = 10)]
            }
        },
        'LogisticRegression': {
            'model': LogisticRegression(random_state=models_tuning_seed, max_iter=1000),
            'params': {
                'penalty': ['l1', 'l2'],
                'C' : [0.001, 0.01, 0.1, 1],
                'solver': ['newton-cg', 'lbfgs', 'sag', 'saga'],
            }
        },
        'MLPClassifier': {
            'model': MLPClassifier(hidden_layer_sizes=(100,100,), random_state=models_tuning_seed, max_iter=1000),
            'params': {
                'activation': ['logistic', 'tanh', 'relu'],
                'solver': ['lbfgs', 'sgd', 'adam'],
                'learning_rate': ['constant', 'invscaling', 'adaptive']
            }
        }
    }
