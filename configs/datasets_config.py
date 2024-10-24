import pathlib

from virny.datasets import (GermanCreditDataset, BankMarketingDataset, CardiovascularDiseaseDataset, DiabetesDataset2019,
                            LawSchoolDataset, ACSIncomeDataset, ACSEmploymentDataset)
from configs.constants import (GERMAN_CREDIT_DATASET, BANK_MARKETING_DATASET, CARDIOVASCULAR_DISEASE_DATASET,
                               DIABETES_DATASET, LAW_SCHOOL_DATASET, ACS_INCOME_DATASET, ACS_EMPLOYMENT_DATASET)


DATASET_CONFIG = {
    GERMAN_CREDIT_DATASET: {
        "data_loader": GermanCreditDataset,
        "data_loader_kwargs": {},
        "test_set_fraction": 0.3,
        "virny_config_path": pathlib.Path(__file__).parent.joinpath('yaml_files', 'german_config.yaml')
    },
    BANK_MARKETING_DATASET: {
        "data_loader": BankMarketingDataset,
        "data_loader_kwargs": {},
        "test_set_fraction": 0.2,
        "virny_config_path": pathlib.Path(__file__).parent.joinpath('yaml_files', 'bank_config.yaml')
    },
    CARDIOVASCULAR_DISEASE_DATASET: {
        "data_loader": CardiovascularDiseaseDataset,
        "data_loader_kwargs": {},
        "test_set_fraction": 0.2,
        "virny_config_path": pathlib.Path(__file__).parent.joinpath('yaml_files', 'heart_config.yaml')
    },
    DIABETES_DATASET: {
        "data_loader": DiabetesDataset2019,
        "data_loader_kwargs": {'with_nulls': False},
        "test_set_fraction": 0.3,
        "virny_config_path": pathlib.Path(__file__).parent.joinpath('yaml_files', 'diabetes_config.yaml')
    },
    LAW_SCHOOL_DATASET: {
        "data_loader": LawSchoolDataset,
        "data_loader_kwargs": {},
        "test_set_fraction": 0.2,
        "virny_config_path": pathlib.Path(__file__).parent.joinpath('yaml_files', 'law_school_config.yaml')
    },
    ACS_INCOME_DATASET: {
        "data_loader": ACSIncomeDataset,
        "data_loader_kwargs": {"state": ['GA'], "year": 2018, "with_nulls": False,
                               "subsample_size": 15_000, "subsample_seed": 42},
        "test_set_fraction": 0.2,
        "virny_config_path": pathlib.Path(__file__).parent.joinpath('yaml_files', 'folk_config.yaml')
    },
    ACS_EMPLOYMENT_DATASET: {
        "data_loader": ACSEmploymentDataset,
        "data_loader_kwargs": {"state": ['CA'], "year": 2018, "with_nulls": False},
        "test_set_fraction": 0.2,
        "virny_config_path": pathlib.Path(__file__).parent.joinpath('yaml_files', 'folk_emp_config.yaml')
    },
}
