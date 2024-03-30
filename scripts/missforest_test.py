import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from virny.datasets.data_loaders import ACSIncomeDataset

module_path = os.path.abspath(os.path.join('.'))
if module_path not in sys.path:
    sys.path.append(module_path)
    
from error_injectors.random_nulls_injector_strategies_v2 import RandomNullsInjectorStrategies
from cleaners.missforest import MissForest
    

data_loader = ACSIncomeDataset(state=['GA'], year=2018, with_nulls=False,
                               subsample_size=20_000, subsample_seed=42)

mcar_injector = RandomNullsInjectorStrategies(seed=42, strategy='MCAR', columns_with_nulls=["AGEP", "SCHL", "MAR"], null_percentage=0.3)
injected_df = mcar_injector.transform(data_loader.X_data)

imputer = MissForest()
repaired_array = imputer.fit_transform(injected_df)

repaired_df = pd.DataFrame(repaired_array, columns=injected_df.columns)

print(repaired_df.isnull().mean() * 100)
