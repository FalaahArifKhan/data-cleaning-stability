from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder


def get_simple_preprocessor(base_flow_dataset):
    cat_ordinal_columns, ordered_categories_lst = zip(*base_flow_dataset.ordered_categories_dct.items())
    cat_ordinal_columns, ordered_categories_lst = list(cat_ordinal_columns), list(ordered_categories_lst)
    cat_nominal_columns = [col for col in base_flow_dataset.categorical_columns if col not in cat_ordinal_columns]

    print('cat_ordinal_columns --', cat_ordinal_columns)
    print('cat_nominal_columns --', cat_nominal_columns)
    print('ordered_categories_lst --', ordered_categories_lst)

    return ColumnTransformer(transformers=[
        ('cat', OneHotEncoder(categories='auto', handle_unknown='ignore', sparse=False), cat_nominal_columns),
        ('num', StandardScaler(), base_flow_dataset.numerical_columns),
        ('ord', OrdinalEncoder(categories=ordered_categories_lst), cat_ordinal_columns)
    ])
