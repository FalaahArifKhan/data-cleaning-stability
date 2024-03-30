from configs.constants import (GERMAN_CREDIT_DATASET, BANK_MARKETING_DATASET, CARDIOVASCULAR_DISEASE_DATASET,
                               DIABETES_DATASET, LAW_SCHOOL_DATASET, ACS_INCOME_DATASET)


EVALUATION_SCENARIOS_CONFIG = {
    ACS_INCOME_DATASET: {
        "MCAR": [
            {
                'missing_features': ['AGEP', 'SCHL', 'MAR', 'COW'],
                'setting': {'error_rates': [0.1, 0.5, 0.9]},
            },
        ],
        "MAR": [
            {
                'missing_features': ['AGEP', 'MAR'],
                'setting': {'condition': ('SEX', '2'), 'error_rates': [0.1, 0.5, 0.9]},
            },
            {
                'missing_features': ['SCHL', 'COW'],
                'setting': {'condition': ('RAC1P', ['2', '3', '4', '5', '6', '7', '8', '9']), 'error_rates': [0.1, 0.5, 0.9]},
            },
        ],
        "MNAR": [
            {
                'missing_features': ['MAR'],
                'setting': {'condition': ('MAR', '5'), 'error_rates': [0.1, 0.5, 0.9]},
            },
            {
                'missing_features': ['COW'],
                'setting': {'condition': ('COW', '9'), 'error_rates': [0.1, 0.5, 0.9]},
            },
            {
                'missing_features': ['AGEP'],
                'setting': {'condition': ('AGEP', [i for i in range(46, 100)]), 'error_rates': [0.1, 0.5, 0.9]},
            },
            {
                'missing_features': ['SCHL'],
                'setting': {'condition': ('SCHL', [str(i) for i in range(1, 15)]), 'error_rates': [0.1, 0.5, 0.9]},
            },
        ],
    },
}
