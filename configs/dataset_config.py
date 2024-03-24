from virny.datasets import (GermanCreditDataset, BankMarketingDataset, CardiovascularDiseaseDataset, DiabetesDataset2019,
                            LawSchoolDataset, ACSIncomeDataset)
from configs.constants import (GERMAN_CREDIT_DATASET, BANK_MARKETING_DATASET, CARDIOVASCULAR_DISEASE_DATASET,
                               DIABETES_DATASET, LAW_SCHOOL_DATASET, ACS_INCOME_DATASET)


DATASET_CONFIG = {
    GERMAN_CREDIT_DATASET: {
        "data_loader": GermanCreditDataset,
        "data_loader_kwargs": {},
        "test_set_fraction": 0.3
    },
    BANK_MARKETING_DATASET: {
        "data_loader": BankMarketingDataset,
        "data_loader_kwargs": {},
        "test_set_fraction": 0.2
    },
    CARDIOVASCULAR_DISEASE_DATASET: {
        "data_loader": CardiovascularDiseaseDataset,
        "data_loader_kwargs": {},
        "test_set_fraction": 0.2
    },
    DIABETES_DATASET: {
        "data_loader": DiabetesDataset2019,
        "data_loader_kwargs": {},
        "test_set_fraction": 0.3
    },
    LAW_SCHOOL_DATASET: {
        "data_loader": LawSchoolDataset,
        "data_loader_kwargs": {},
        "test_set_fraction": 0.2
    },
    ACS_INCOME_DATASET: {
        "data_loader": ACSIncomeDataset,
        "data_loader_kwargs": {"state": ['GA'], "year": 2018, "with_nulls": False,
                               "subsample_size": 15_000, "subsample_seed": 42},
        "test_set_fraction": 0.2
    },
}
