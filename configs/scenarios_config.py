from configs.constants import (GERMAN_CREDIT_DATASET, BANK_MARKETING_DATASET, CARDIOVASCULAR_DISEASE_DATASET,
                               DIABETES_DATASET, LAW_SCHOOL_DATASET, ACS_INCOME_DATASET)


EVALUATION_SCENARIOS_CONFIG = {
    'exp3_mcar3': {
        'train_injection_scenario': 'MCAR3',
        'test_injection_scenarios': ['MCAR2', 'MAR2', 'MNAR2'],
    },
    'exp3_mar3': {
        'train_injection_scenario': 'MAR3',
        'test_injection_scenarios': ['MCAR2', 'MAR2', 'MNAR2'],
    },
    'exp3_mnar3': {
        'train_injection_scenario': 'MNAR3',
        'test_injection_scenarios': ['MCAR2', 'MAR2', 'MNAR2'],
    },
    'exp1&2_mcar1': {
        'train_injection_scenario': 'MCAR1',
        'test_injection_scenarios': ['MCAR2', 'MAR2', 'MNAR2'],
    },
    'exp1&2_mcar5': {
        'train_injection_scenario': 'MCAR5',
        'test_injection_scenarios': ['MCAR2', 'MAR2', 'MNAR2'],
    },
    'exp1&2_mar1': {
        'train_injection_scenario': 'MAR1',
        'test_injection_scenarios': ['MCAR2', 'MAR2', 'MNAR2'],
    },
    'exp1&2_mar5': {
        'train_injection_scenario': 'MAR5',
        'test_injection_scenarios': ['MCAR2', 'MAR2', 'MNAR2'],
    },
    'exp1&2_mnar1': {
        'train_injection_scenario': 'MNAR1',
        'test_injection_scenarios': ['MCAR2', 'MAR2', 'MNAR2'],
    },
    'exp1&2_mnar5': {
        'train_injection_scenario': 'MNAR5',
        'test_injection_scenarios': ['MCAR2', 'MAR2', 'MNAR2'],
    },
    'exp1&2_mcar3': {
        'train_injection_scenario': 'MCAR3',
        'test_injection_scenarios': [
            'MCAR1', 'MAR1', 'MNAR1',
            'MCAR2', 'MAR2', 'MNAR2',
            'MCAR3', 'MAR3', 'MNAR3',
            'MCAR4', 'MAR4', 'MNAR4',
            'MCAR5', 'MAR5', 'MNAR5',
        ],
    },
    'exp1&2_mar3': {
        'train_injection_scenario': 'MAR3',
        'test_injection_scenarios': [
            'MCAR1', 'MAR1', 'MNAR1',
            'MCAR2', 'MAR2', 'MNAR2',
            'MCAR3', 'MAR3', 'MNAR3',
            'MCAR4', 'MAR4', 'MNAR4',
            'MCAR5', 'MAR5', 'MNAR5',
        ],
    },
    'exp1&2_mnar3': {
        'train_injection_scenario': 'MNAR3',
        'test_injection_scenarios': [
            'MCAR1', 'MAR1', 'MNAR1',
            'MCAR2', 'MAR2', 'MNAR2',
            'MCAR3', 'MAR3', 'MNAR3',
            'MCAR4', 'MAR4', 'MNAR4',
            'MCAR5', 'MAR5', 'MNAR5',
        ],
    },
}

ERROR_INJECTION_SCENARIOS_CONFIG = {
    ACS_INCOME_DATASET: {
        "MCAR": [
            {
                'missing_features': ['WKHP', 'AGEP', 'SCHL', 'MAR'],
                'setting': {'error_rates': [0.1, 0.2, 0.3, 0.4, 0.5]},
            },
        ],
        "MAR": [
            {
                'missing_features': ['WKHP', 'SCHL'],
                'setting': {'condition': ('SEX', '2'), 'error_rates': [0.08, 0.12, 0.20, 0.28, 0.35]},
            },
            {
                'missing_features': ['WKHP', 'SCHL'],
                'setting': {'condition': ('SEX', '1'), 'error_rates': [0.02, 0.08, 0.10, 0.12, 0.15]},
            },
            {
                'missing_features': ['MAR', 'AGEP'],
                'setting': {'condition': ('RAC1P', ['2', '3', '4', '5', '6', '7', '8', '9']), 'error_rates': [0.08, 0.12, 0.20, 0.28, 0.35]},
            },
            {
                'missing_features': ['MAR', 'AGEP'],
                'setting': {'condition': ('RAC1P', '1'), 'error_rates': [0.02, 0.08, 0.10, 0.12, 0.15]},
            },
        ],
        "MNAR": [
            {
                'missing_features': ['MAR'],
                'setting': {'condition': ('MAR', '1'), 'error_rates': [0.01, 0.02, 0.05, 0.10, 0.15]},
            },
            {
                'missing_features': ['MAR'],
                'setting': {'condition': ('MAR', ['2', '3', '4', '5']), 'error_rates': [0.09, 0.18, 0.25, 0.30, 0.35]},
            },
            {
                'missing_features': ['WKHP'],  # TODO: check
                'setting': {'condition': ('WKHP', [i * 1.0 for i in range(1, 40)]), 'error_rates': [0.09, 0.18, 0.25, 0.30, 0.35]},
            },
            {
                'missing_features': ['WKHP'],
                'setting': {'condition': ('WKHP', [i * 1.0 for i in range(40, 101)]), 'error_rates': [0.01, 0.02, 0.05, 0.10, 0.15]},
            },
            {
                'missing_features': ['AGEP'],
                'setting': {'condition': ('AGEP', [i for i in range(17, 51)]), 'error_rates': [0.01, 0.02, 0.05, 0.10, 0.15]},
            },
            {
                'missing_features': ['AGEP'],
                'setting': {'condition': ('AGEP', [i for i in range(51, 100)]), 'error_rates': [0.09, 0.18, 0.25, 0.30, 0.35]},
            },
            {
                'missing_features': ['SCHL'],
                'setting': {'condition': ('SCHL', [str(i) for i in range(1, 21)]), 'error_rates': [0.09, 0.18, 0.25, 0.30, 0.35]},
            },
            {
                'missing_features': ['SCHL'],
                'setting': {'condition': ('SCHL', [str(i) for i in range(21, 25)]), 'error_rates': [0.01, 0.02, 0.05, 0.10, 0.15]},
            },
        ],
    },
}
