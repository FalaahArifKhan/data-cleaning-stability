from configs.constants import (GERMAN_CREDIT_DATASET, BANK_MARKETING_DATASET, CARDIOVASCULAR_DISEASE_DATASET,
                               DIABETES_DATASET, LAW_SCHOOL_DATASET, ACS_INCOME_DATASET)


EVALUATION_SCENARIOS_CONFIG = {
    'exp1_mcar3': {
        'train_injection_scenario': 'MCAR3',
        'test_injection_scenarios': ['MCAR3', 'MAR3', 'MNAR3'],
    },
    'exp1_mar3': {
        'train_injection_scenario': 'MAR3',
        'test_injection_scenarios': ['MCAR3', 'MAR3', 'MNAR3'],
    },
    'exp1_mnar3': {
        'train_injection_scenario': 'MNAR3',
        'test_injection_scenarios': ['MCAR3', 'MAR3', 'MNAR3'],
    },
    'exp2&3_mcar1': {
        'train_injection_scenario': 'MCAR1',
        'test_injection_scenarios': ['MCAR3', 'MAR3', 'MNAR3'],
    },
    'exp2&3_mcar5': {
        'train_injection_scenario': 'MCAR5',
        'test_injection_scenarios': ['MCAR3', 'MAR3', 'MNAR3'],
    },
    'exp2&3_mar1': {
        'train_injection_scenario': 'MAR1',
        'test_injection_scenarios': ['MCAR3', 'MAR3', 'MNAR3'],
    },
    'exp2&3_mar5': {
        'train_injection_scenario': 'MAR5',
        'test_injection_scenarios': ['MCAR3', 'MAR3', 'MNAR3'],
    },
    'exp2&3_mnar1': {
        'train_injection_scenario': 'MNAR1',
        'test_injection_scenarios': ['MCAR3', 'MAR3', 'MNAR3'],
    },
    'exp2&3_mnar5': {
        'train_injection_scenario': 'MNAR5',
        'test_injection_scenarios': ['MCAR3', 'MAR3', 'MNAR3'],
    },
    'exp2&3_mcar3': {
        'train_injection_scenario': 'MCAR3',
        'test_injection_scenarios': [
            'MCAR1', 'MAR1', 'MNAR1',
            'MCAR2', 'MAR2', 'MNAR2',
            'MCAR3', 'MAR3', 'MNAR3',
            'MCAR4', 'MAR4', 'MNAR4',
            'MCAR5', 'MAR5', 'MNAR5',
        ],
    },
    'exp2&3_mar3': {
        'train_injection_scenario': 'MAR3',
        'test_injection_scenarios': [
            'MCAR1', 'MAR1', 'MNAR1',
            'MCAR2', 'MAR2', 'MNAR2',
            'MCAR3', 'MAR3', 'MNAR3',
            'MCAR4', 'MAR4', 'MNAR4',
            'MCAR5', 'MAR5', 'MNAR5',
        ],
    },
    'exp2&3_mnar3': {
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
                'missing_features': ['WKHP'],
                'setting': {'condition': ('WKHP', {'lt': 40.0}), 'error_rates': [0.09, 0.18, 0.25, 0.30, 0.35]},
            },
            {
                'missing_features': ['WKHP'],
                'setting': {'condition': ('WKHP', {'ge': 40.0}), 'error_rates': [0.01, 0.02, 0.05, 0.10, 0.15]},
            },
            {
                'missing_features': ['AGEP'],
                'setting': {'condition': ('AGEP', {'le': 50}), 'error_rates': [0.01, 0.02, 0.05, 0.10, 0.15]},
            },
            {
                'missing_features': ['AGEP'],
                'setting': {'condition': ('AGEP', {'gt': 50}), 'error_rates': [0.09, 0.18, 0.25, 0.30, 0.35]},
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
    LAW_SCHOOL_DATASET: {
        "MCAR": [
            {
                'missing_features': ['zfygpa', 'ugpa', 'fam_inc', 'tier'],
                'setting': {'error_rates': [0.1, 0.2, 0.3, 0.4, 0.5]},
            },
        ],
        "MAR": [
            {
                'missing_features': ['ugpa', 'zfygpa'],
                'setting': {'condition': ('male', '1'), 'error_rates': [0.02, 0.08, 0.10, 0.12, 0.15]},
            },
            {
                'missing_features': ['ugpa', 'zfygpa'],
                'setting': {'condition': ('male', '0'), 'error_rates': [0.08, 0.12, 0.20, 0.28, 0.35]},
            },
            {
                'missing_features': ['fam_inc', 'tier'],
                'setting': {'condition': ('race', 'White'), 'error_rates': [0.02, 0.08, 0.15, 0.25, 0.35]},
            },
            {
                'missing_features': ['fam_inc', 'tier'],
                'setting': {'condition': ('race', 'Non-White'), 'error_rates': [0.08, 0.12, 0.15, 0.15, 0.15]},
            },
        ],
        "MNAR": [
            {
                'missing_features': ['ugpa'],
                'setting': {'condition': ('ugpa', {'ge': 3.0}), 'error_rates': [0.02, 0.08, 0.10, 0.12, 0.15]},
            },
            {
                'missing_features': ['ugpa'],
                'setting': {'condition': ('ugpa', {'lt': 3.0}), 'error_rates': [0.08, 0.12, 0.20, 0.28, 0.35]},
            },
            {
                'missing_features': ['zfygpa'],
                'setting': {'condition': ('zfygpa', {'gt': 0.0}), 'error_rates': [0.02, 0.08, 0.10, 0.12, 0.15]},
            },
            {
                'missing_features': ['zfygpa'],
                'setting': {'condition': ('zfygpa', {'le': 0.0}), 'error_rates': [0.08, 0.12, 0.20, 0.28, 0.35]},
            },
            {
                'missing_features': ['fam_inc'],
                'setting': {'condition': ('fam_inc', ['4', '5']), 'error_rates': [0.02, 0.08, 0.10, 0.12, 0.15]},
            },
            {
                'missing_features': ['fam_inc'],
                'setting': {'condition': ('fam_inc', ['1', '2', '3']), 'error_rates': [0.08, 0.12, 0.20, 0.28, 0.35]},
            },
            {
                'missing_features': ['tier'],
                'setting': {'condition': ('tier', ['4', '5', '6']), 'error_rates': [0.02, 0.08, 0.10, 0.12, 0.15]},
            },
            {
                'missing_features': ['tier'],
                'setting': {'condition': ('tier', ['1', '2', '3']), 'error_rates': [0.08, 0.12, 0.20, 0.28, 0.35]},
            },
        ]
    },
    GERMAN_CREDIT_DATASET: {
        "MCAR": [
            {
                'missing_features': ['duration', 'credit-amount', 'checking-account',
                                     'savings-account', 'employment-since'],
                'setting': {'error_rates': [0.1, 0.2, 0.3, 0.4, 0.5]},
            },
        ],
        "MAR": [
            {
                'missing_features': ['savings-account', 'checking-account', 'credit-amount'],
                'setting': {'condition': ('age', {'le': 25}), 'error_rates': [0.08, 0.12, 0.18, 0.18, 0.18]},
            },
            {
                'missing_features': ['savings-account', 'checking-account', 'credit-amount'],
                'setting': {'condition': ('age', {'gt': 25}), 'error_rates': [0.02, 0.08, 0.12, 0.22, 0.32]},
            },
            {
                'missing_features': ['employment-since', 'duration'],
                'setting': {'condition': ('sex', 'female'), 'error_rates': [0.08, 0.12, 0.20, 0.28, 0.3]},
            },
            {
                'missing_features': ['employment-since', 'duration'],
                'setting': {'condition': ('sex', 'male'), 'error_rates': [0.02, 0.08, 0.10, 0.12, 0.2]},
            },
        ],
        "MNAR": [
            {
                'missing_features': ['checking-account'],
                'setting': {'condition': ('checking-account', 'no account'), 'error_rates': [0.09, 0.18, 0.25, 0.30, 0.35]},
            },
            {
                'missing_features': ['checking-account'],
                'setting': {'condition': ('checking-account', ['<0 DM', '0 <= <200 DM', '>= 200 DM']), 'error_rates': [0.01, 0.02, 0.05, 0.10, 0.15]},
            },
            {
                'missing_features': ['duration'],
                'setting': {'condition': ('duration', {'le': 20}), 'error_rates': [0.09, 0.18, 0.25, 0.30, 0.35]},
            },
            {
                'missing_features': ['duration'],
                'setting': {'condition': ('duration', {'gt': 20}), 'error_rates': [0.01, 0.02, 0.05, 0.10, 0.15]},
            },
            {
                'missing_features': ['savings-account'],
                'setting': {'condition': ('savings-account', 'no savings account'), 'error_rates': [0.08, 0.12, 0.10, 0.15, 0.15]},
            },
            {
                'missing_features': ['savings-account'],
                'setting': {'condition': ('savings-account', ['<100 DM', '500 <= < 1000 DM', '>= 1000 DM', '100 <= <500 DM']), 'error_rates': [0.02, 0.08, 0.20, 0.25, 0.35]},
            },
            {
                'missing_features': ['employment-since'],
                'setting': {'condition': ('employment-since', ['<1 years', 'unemployed']), 'error_rates': [0.09, 0.18, 0.20, 0.20, 0.20]},
            },
            {
                'missing_features': ['employment-since'],
                'setting': {'condition': ('employment-since', ['1<= < 4 years', '4<= <7 years', '>=7 years']), 'error_rates': [0.01, 0.02, 0.10, 0.20, 0.30]},
            },
            {
                'missing_features': ['credit-amount'],
                'setting': {'condition': ('credit-amount', {'gt': 5000}), 'error_rates': [0.09, 0.18, 0.25, 0.30, 0.35]},
            },
            {
                'missing_features': ['credit-amount'],
                'setting': {'condition': ('credit-amount', {'le': 5000}), 'error_rates': [0.01, 0.02, 0.05, 0.10, 0.15]},
            },
        ],
    },
    DIABETES_DATASET: {
        "MCAR": [
            {
                'missing_features': ['SoundSleep', 'Family_Diabetes', 'PhysicallyActive', 'RegularMedicine'],
                'setting': {'error_rates': [0.1, 0.2, 0.3, 0.4, 0.5]},
            },
        ],
        "MAR": [
            {
                'missing_features': ['Family_Diabetes', 'RegularMedicine'],
                'setting': {'condition': ('Gender', 'Female'), 'error_rates': [0.08, 0.15, 0.20, 0.25, 0.30]},
            },
            {
                'missing_features': ['Family_Diabetes', 'RegularMedicine'],
                'setting': {'condition': ('Gender', 'Male'), 'error_rates': [0.02, 0.05, 0.10, 0.15, 0.20]},
            },
            {
                'missing_features': ['PhysicallyActive', 'SoundSleep'],
                'setting': {'condition': ('Age', 'less than 40'), 'error_rates': [0.02, 0.05, 0.10, 0.15, 0.20]},
            },
            {
                'missing_features': ['PhysicallyActive', 'SoundSleep'],
                'setting': {'condition': ('Age', ['40-49', '50-59', '60 or older']), 'error_rates': [0.08, 0.15, 0.20, 0.25, 0.30]},
            },
        ],
        "MNAR": [
            {
                'missing_features': ['Family_Diabetes'],
                'setting': {'condition': ('Family_Diabetes', 'yes'), 'error_rates': [0.09, 0.18, 0.25, 0.30, 0.35]},
            },
            {
                'missing_features': ['Family_Diabetes'],
                'setting': {'condition': ('Family_Diabetes', 'no'), 'error_rates': [0.01, 0.02, 0.05, 0.10, 0.15]},
            },
            {
                'missing_features': ['RegularMedicine'],
                'setting': {'condition': ('RegularMedicine', 'yes'), 'error_rates': [0.08, 0.15, 0.20, 0.25, 0.30]},
            },
            {
                'missing_features': ['RegularMedicine'],
                'setting': {'condition': ('RegularMedicine', 'no'), 'error_rates': [0.02, 0.05, 0.10, 0.15, 0.20]},
            },
            {
                'missing_features': ['PhysicallyActive'],
                'setting': {'condition': ('PhysicallyActive', ['none', 'less than half an hr']), 'error_rates': [0.09, 0.18, 0.25, 0.30, 0.35]},
            },
            {
                'missing_features': ['PhysicallyActive'],
                'setting': {'condition': ('PhysicallyActive', ['one hr or more', 'more than half an hr']), 'error_rates': [0.01, 0.02, 0.05, 0.10, 0.15]},
            },
            {
                'missing_features': ['SoundSleep'],
                'setting': {'condition': ('SoundSleep', {'lt': 5}), 'error_rates': [0.08, 0.15, 0.20, 0.25, 0.30]},
            },
            {
                'missing_features': ['SoundSleep'],
                'setting': {'condition': ('SoundSleep', {'ge': 5}), 'error_rates': [0.02, 0.05, 0.10, 0.15, 0.20]},
            },
        ],
    }
}
