import os
import sys
import numpy as np
import pandas as pd

module_path = os.path.abspath(os.path.join('.'))
if module_path not in sys.path:
    sys.path.append(module_path)
    
from virny.datasets.data_loaders import ACSIncomeDataset
from error_injectors.random_nulls_injector_strategies_v2 import RandomNullsInjectorStrategies


if __name__ == "__main__":
    
    data_loader = ACSIncomeDataset(state=['GA'], year=2018, with_nulls=False,
                                subsample_size=20_000, subsample_seed=42)

    print(data_loader.X_data.head())  

    mcar_injector = RandomNullsInjectorStrategies(seed=42, strategy='MCAR', columns_with_nulls=["AGEP", "SCHL", "MAR"], null_percentage=0.4)
    df_injected_mcar = mcar_injector.transform(data_loader.X_data)

    mcar_injector2 = RandomNullsInjectorStrategies(seed=42, strategy='MCAR', columns_with_nulls=["AGEP", "SCHL", "MAR"], null_percentage=0.4)
    df_injected_mcar2 = mcar_injector2.transform(data_loader.X_data)

    # print percentage of nulls in each column
    print("\nMCAR strategy")
    print(df_injected_mcar.isnull().mean() * 100)
    print(f"Same results with 1 seed: {df_injected_mcar.equals(df_injected_mcar2)}")

    mar_injector = RandomNullsInjectorStrategies(seed=42, strategy='MAR', columns_with_nulls=["AGEP", "MAR"], null_percentage=0.4, condition=("SEX", "2"))
    df_injected_mar = mar_injector.transform(data_loader.X_data)

    mar_injector2 = RandomNullsInjectorStrategies(seed=42, strategy='MAR', columns_with_nulls=["AGEP", "MAR"], null_percentage=0.4, condition=("SEX", "2"))
    df_injected_mar2 = mar_injector2.transform(data_loader.X_data)

    # print percentage of nulls in each column
    print("\nMAR strategy")
    print(df_injected_mar[df_injected_mar["SEX"] == "2"].isnull().mean() * 100)
    print(f"Same results with 1 seed: {df_injected_mar.equals(df_injected_mar2)}")

    mnar_injector = RandomNullsInjectorStrategies(seed=42, strategy='MNAR', columns_with_nulls=["SEX"], null_percentage=0.4, condition=("SEX", "2"))
    df_injected_mnar = mnar_injector.transform(data_loader.X_data)

    mnar_injector = RandomNullsInjectorStrategies(seed=42, strategy='MNAR', columns_with_nulls=["SEX"], null_percentage=0.4, condition=("SEX", "2"))
    df_injected_mnar2 = mnar_injector.transform(data_loader.X_data)

    # print percentage of nulls in each column
    print("MNAR strategy")
    print(df_injected_mnar2[data_loader.X_data["SEX"] == "2"].isnull().mean() * 100)
    print(f"Same results with 1 seed: {df_injected_mnar.equals(df_injected_mnar2)}")