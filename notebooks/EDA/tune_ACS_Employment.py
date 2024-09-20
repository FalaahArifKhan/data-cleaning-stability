import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')
os.environ["PYTHONWARNINGS"] = "ignore"

# Define a correct root path
sys.path.append(str(Path(f"{__file__}").parent.parent))
print('Current location: ', os.getcwd())

from datetime import datetime, timezone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from virny.datasets import ACSEmploymentDataset
from virny.utils.model_tuning_utils import tune_ML_models
from virny.preprocessing.basic_preprocessing import preprocess_dataset


if __name__ == '__main__':
    DATASET_SPLIT_SEED = 42
    MODELS_TUNING_SEED = 42
    TEST_SET_FRACTION = 0.2
    DATASET_NAME = 'ACS_Employment_CA_2018'

    sensitive_attributes_dct = {'SEX': '2', 'RAC1P': ['2', '3', '4', '5', '6', '7', '8', '9'], 'SEX & RAC1P': None}
    sensitive_attributes = [attr for attr in sensitive_attributes_dct.keys() if '&' not in attr]

    data_loader = ACSEmploymentDataset(state=['CA'], year=2018, with_nulls=False)

    column_transformer = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(categories='auto', handle_unknown='ignore', sparse=False), data_loader.categorical_columns),
            ('num', StandardScaler(), data_loader.numerical_columns),
        ],
        verbose_feature_names_out=False
    )
    base_flow_dataset = preprocess_dataset(data_loader=data_loader,
                                           column_transformer=column_transformer,
                                           sensitive_attributes_dct=sensitive_attributes_dct,
                                           test_set_fraction=TEST_SET_FRACTION,
                                           dataset_split_seed=DATASET_SPLIT_SEED)

    models_params_for_tuning = {
        'RandomForestClassifier': {
            'model': RandomForestClassifier(random_state=MODELS_TUNING_SEED),
            'params': {
                'n_estimators': [100, 200, 500],
                'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'bootstrap': [True, False]
            }
        },
    }
    tuned_params_df, models_config = tune_ML_models(models_params_for_tuning, base_flow_dataset, DATASET_NAME, n_folds=3)
    print(tuned_params_df)

    now = datetime.now(timezone.utc)
    date_time_str = now.strftime("%Y%m%d__%H%M%S")
    tuned_df_path = os.path.join(os.getcwd(), 'models_tuning', f'tuning_results_{DATASET_NAME}.csv')
    tuned_params_df.to_csv(tuned_df_path, sep=",", columns=tuned_params_df.columns, float_format="%.4f", index=False)
