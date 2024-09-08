from tabpfn import TabPFNClassifier


class TabPFNClassifierWrapper(TabPFNClassifier):
    def set_params(self, **params):
        if 'random_state' in params.keys():
            params['seed'] = params.pop('random_state')

        super().set_params(**params)
        return self
