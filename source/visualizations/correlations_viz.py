import pandas as pd

from configs.constants import (DIABETES_DATASET, GERMAN_CREDIT_DATASET, BANK_MARKETING_DATASET,
                               CARDIOVASCULAR_DISEASE_DATASET, ACS_INCOME_DATASET, LAW_SCHOOL_DATASET,
                               ACS_EMPLOYMENT_DATASET)
from configs.scenarios_config import EVALUATION_SCENARIOS_CONFIG

from source.visualizations.model_metrics_extraction_for_viz import (
    get_evaluation_scenario, get_models_metric_df, get_overall_metric_from_disparity_metric,
    get_models_disparity_metric_df, get_baseline_model_median, get_base_rate)


def get_data_for_correlation_plots(db_client_1, db_client_3, dataset_names: list, missingness_types: list,
                                   metric_names: list, dataset_to_group: dict = None):
    missingness_type_to_train_and_test_sets = {
        'single_mechanism': [
            {'train': 'MCAR3', 'test': 'MCAR3'},
            {'train': 'MAR3', 'test': 'MAR3'},
            {'train': 'MNAR3', 'test': 'MNAR3'},
        ],
        'multi_mechanism': [
            {'train': 'mixed_exp', 'test': 'MCAR1 & MAR1 & MNAR1'},
        ],
        'missingness_shift': [
            {'train': 'MCAR3', 'test': 'MAR3'},
            {'train': 'MCAR3', 'test': 'MNAR3'},
            {'train': 'MAR3', 'test': 'MCAR3'},
            {'train': 'MAR3', 'test': 'MNAR3'},
            {'train': 'MNAR3', 'test': 'MCAR3'},
            {'train': 'MNAR3', 'test': 'MAR3'},
        ],
    }
    metrics_df = pd.DataFrame()
    for metric_name in metric_names:
        metric_name = '_'.join([c.capitalize() for c in metric_name.split('_')]) if 'equalized_odds' not in metric_name.lower() else metric_name

        for missingness_type in missingness_types:
            db_client = db_client_3 if missingness_type == 'multi_mechanism' else db_client_1

            for injection_scenarios in missingness_type_to_train_and_test_sets[missingness_type]:
                train_injection_scenario, test_injection_scenario = injection_scenarios['train'], injection_scenarios['test']
                selected_dataset_to_group = dataset_to_group if get_overall_metric_from_disparity_metric(disparity_metric=metric_name) is not None else None
                missingness_type_df = get_data_for_correlation_plots_for_diff_imputers_and_datasets(db_client=db_client,
                                                                                                    dataset_names=dataset_names,
                                                                                                    train_injection_scenario=train_injection_scenario,
                                                                                                    test_injection_scenario=test_injection_scenario,
                                                                                                    metric_name=metric_name,
                                                                                                    dataset_to_group=selected_dataset_to_group)
                missingness_type_df['Missingness_Type'] = missingness_type
                missingness_type_df['Train_Injection_Scenario'] = train_injection_scenario
                metrics_df = pd.concat([metrics_df, missingness_type_df])

            print(f'Extracted data for "{metric_name}" metric and "{missingness_type}" missingness type')

    return metrics_df


def get_data_for_correlation_plots_for_diff_imputers_and_datasets(db_client, dataset_names: list,
                                                                  train_injection_scenario: str, test_injection_scenario: str,
                                                                  metric_name: str, dataset_to_group: dict = None):
    evaluation_scenario = get_evaluation_scenario(train_injection_scenario)

    models_metric_df_for_diff_datasets = pd.DataFrame()
    for dataset_name in dataset_names:
        group = 'overall' if dataset_to_group is None else dataset_to_group[dataset_name]

        if group == 'overall':
            models_metric_df = get_models_metric_df(db_client=db_client,
                                                    dataset_name=dataset_name,
                                                    evaluation_scenario=evaluation_scenario,
                                                    metric_name=metric_name,
                                                    group=group)
        else:
            overall_metric = get_overall_metric_from_disparity_metric(disparity_metric=metric_name)
            models_metric_df = get_models_disparity_metric_df(db_client=db_client,
                                                              dataset_name=dataset_name,
                                                              evaluation_scenario=evaluation_scenario,
                                                              metric_name=overall_metric,
                                                              group=group)
            models_metric_df['Dataset_Name'] = dataset_name

        models_metric_df = models_metric_df[models_metric_df['Metric'] == metric_name]
        models_metric_df = models_metric_df.rename(columns={group: 'Metric_Value'})

        models_metric_df['Test_Injection_Scenario'] = models_metric_df.apply(
            lambda row: EVALUATION_SCENARIOS_CONFIG[row['Evaluation_Scenario']]['test_injection_scenarios'][row['Test_Set_Index']],
            axis=1
        )
        models_metric_df = models_metric_df[models_metric_df['Test_Injection_Scenario'] == test_injection_scenario]

        models_metric_df_for_diff_datasets = pd.concat([models_metric_df_for_diff_datasets, models_metric_df])
        # print(f'Extracted data for {dataset_name}')

    return models_metric_df_for_diff_datasets
