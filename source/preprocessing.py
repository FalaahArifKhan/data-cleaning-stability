from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder


def get_simple_preprocessor(base_flow_dataset):
    return ColumnTransformer(transformers=[
        ('cat', OneHotEncoder(categories='auto', handle_unknown='ignore', sparse=False), base_flow_dataset.categorical_columns),
        ('num', StandardScaler(), base_flow_dataset.numerical_columns),
    ])
