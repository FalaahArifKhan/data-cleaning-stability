from configs.constants import (GERMAN_CREDIT_DATASET, BANK_MARKETING_DATASET, CARDIOVASCULAR_DISEASE_DATASET,
                               DIABETES_DATASET, LAW_SCHOOL_DATASET, ACS_INCOME_DATASET)


DATASET_CONFIG = {
    ACS_INCOME_DATASET: {
        "MCAR": [
            {
                'missing_features': ['AGEP', 'SCHL', 'MAR', 'DIS'],
                'error_rates': [0.1, 0.5, 0.9]
            },
        ],
        "MAR": [
            {
                'missing_features': ['AGEP', 'MAR'],
                'conditions': [
                    {'SEX': '2', 'error_rates': [0.1, 0.5, 0.9]},
                    {'SEX': '1', 'error_rates': [0.1, 0.5, 0.9]},
                ]
            },
            {
                'missing_features': ['SCHL', 'DIS'],
                'conditions': [
                    {'RAC1P': '1', 'error_rates': [0.1, 0.5, 0.9]},
                    {'RAC1P': ['2', '3', '4', '5', '6', '7', '8', '9'], 'error_rates': [0.1, 0.5, 0.9]},
                ]
            },
        ],
        "MNAR": [
            {
                'missing_features': ['MAR'],
                'conditions': [
                    {'MAR': '5', 'error_rates': [0.1, 0.5, 0.9]},
                    {'MAR': ['1', '2', '3', '4'], 'error_rates': [0.1, 0.5, 0.9]},
                ]
            },
            {
                'missing_features': ['DIS'],
                'conditions': [
                    {'DIS': '2', 'error_rates': [0.0, 0.1, 0.3]},
                    {'DIS': '1', 'error_rates': [0.1, 0.5, 0.9]},
                ]
            },
            {
                'missing_features': ['AGEP'],
                'conditions': [
                    {'AGEP': [i for i in range(46, 100)], 'error_rates': [0.1, 0.3, 0.9]},
                ]
            },
            {
                'missing_features': ['SCHL'],
                'conditions': [
                    {'SCHL': [str(i) for i in range(1, 15)], 'error_rates': [0.1, 0.3, 0.9]},
                ]
            },
        ],
    },
}
