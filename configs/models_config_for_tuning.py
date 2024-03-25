import numpy as np
from pprint import pprint
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier


def get_models_params_for_tuning(models_tuning_seed):
    return {
        'DecisionTreeClassifier': {
            'model': DecisionTreeClassifier(random_state=models_tuning_seed),
            'params': {
                "max_depth": [5, 10, 20, 30],
                'min_samples_leaf': [5, 10, 20, 50, 100],
                "max_features": [0.6, 'sqrt'],
                "criterion": ["gini", "entropy"]
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
        'LGBMClassifier': {
            'model': LGBMClassifier(random_state=models_tuning_seed),
            'params': {
                'max_depth' : [i for i in range(3, 12)],
                'num_leaves' : [int(x) for x in np.linspace(start = 20, stop = 3000, num = 10)],
                'min_data_in_leaf' : [int(x) for x in np.linspace(start = 100, stop = 1000, num = 10)],
            }
        },
    }


if __name__ == '__main__':
    model_tuning_params = get_models_params_for_tuning(models_tuning_seed=200)
    pprint(model_tuning_params)
