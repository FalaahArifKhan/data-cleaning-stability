from virny.datasets import ACSIncomeDataset
#from missforest_imputer import MissForestImputer
from kmeans_imputer import KMeansImputer
from error_injectors.nulls_injector import NullsInjector
from utils.dataframe_utils import get_object_columns_indexes
import pandas as pd

SEED = 42

# Load the dataset
data_loader = ACSIncomeDataset(state=['GA'], year=2018, with_nulls=False,
                                   subsample_size=5_000, subsample_seed=SEED)
mcar_injector = NullsInjector(seed=SEED, strategy='MCAR', columns_with_nulls=["AGEP", "SCHL", "MAR"], null_percentage=0.3)
injected_df = mcar_injector.transform(data_loader.X_data)

categorical_columns_idxs = get_object_columns_indexes(injected_df)


kmeans_imputer = KMeansImputer(seed=SEED, n_clusters=3)

repaired_array = kmeans_imputer.fit_transform(injected_df, cat_vars=categorical_columns_idxs)
repaired_df = pd.DataFrame(repaired_array, columns=injected_df.columns)
print(repaired_df.isnull().mean() * 100)