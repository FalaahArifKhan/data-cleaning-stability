import pandas as pd
from virny.custom_classes.metrics_composer import MetricsComposer
from virny.configs.constants import *

from configs.constants import (EXP_COLLECTION_NAME, DIABETES_DATASET, GERMAN_CREDIT_DATASET, BANK_MARKETING_DATASET,
                               CARDIOVASCULAR_DISEASE_DATASET, ACS_INCOME_DATASET, LAW_SCHOOL_DATASET)
from configs.scenarios_config import EVALUATION_SCENARIOS_CONFIG
from configs.datasets_config import DATASET_CONFIG
from source.custom_classes.database_client import DatabaseClient


DB_CLIENT_2 = DatabaseClient()
DB_CLIENT_2.connect()


def get_base_rate(dataset_name: str):
    data_loader = DATASET_CONFIG[dataset_name]["data_loader"](
        **DATASET_CONFIG[dataset_name]["data_loader_kwargs"]
    )
    y_data = data_loader.y_data
    overall_base_rate = max(y_data[y_data == 0].shape[0] / y_data.shape[0],
                            y_data[y_data == 1].shape[0] / y_data.shape[0])

    return overall_base_rate


def get_evaluation_scenario(train_injection_scenario):
    evaluation_scenarios = list(EVALUATION_SCENARIOS_CONFIG.keys())
    evaluation_scenarios.remove('exp2_3_mcar3')
    evaluation_scenarios.remove('exp2_3_mar3')
    evaluation_scenarios.remove('exp2_3_mnar3')

    for evaluation_scenario in evaluation_scenarios:
        if train_injection_scenario.lower() in evaluation_scenario:
            return evaluation_scenario

    return None


def get_data_for_box_plots_for_diff_imputers_and_datasets(train_injection_scenario: str, test_injection_scenario: str,
                                                          metric_name: str, db_client, dataset_to_group: dict = None):
    dataset_to_model_name_dct = {
        DIABETES_DATASET: 'rf_clf',
        GERMAN_CREDIT_DATASET: 'rf_clf',
        ACS_INCOME_DATASET: 'mlp_clf',
        LAW_SCHOOL_DATASET: 'lr_clf',
        BANK_MARKETING_DATASET: 'lgbm_clf',
        CARDIOVASCULAR_DISEASE_DATASET: 'lgbm_clf',
    }
    evaluation_scenario = get_evaluation_scenario(train_injection_scenario)

    models_metric_df_for_diff_datasets = pd.DataFrame()
    for dataset_name in DATASET_CONFIG.keys():
        group = 'overall' if dataset_to_group is None else dataset_to_group[dataset_name]
        model_name = dataset_to_model_name_dct[dataset_name]

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

        models_metric_df = models_metric_df[models_metric_df['Model_Name'].isin([model_name, 'boost_clean'])]
        models_metric_df['Test_Injection_Scenario'] = models_metric_df.apply(
            lambda row: EVALUATION_SCENARIOS_CONFIG[row['Evaluation_Scenario']]['test_injection_scenarios'][row['Test_Set_Index']],
            axis=1
        )
        models_metric_df = models_metric_df[models_metric_df['Test_Injection_Scenario'] == test_injection_scenario]

        # Add a baseline median to models_metric_df to display it as a horizontal line
        baseline_median = get_baseline_model_median(dataset_name=dataset_name,
                                                    model_name=model_name,
                                                    metric_name=metric_name,
                                                    db_client=db_client,
                                                    group=group)
        models_metric_df['Baseline_Median'] = baseline_median

        if metric_name == 'Accuracy':
            models_metric_df['Base_Rate'] = get_base_rate(dataset_name)

        models_metric_df_for_diff_datasets = pd.concat([models_metric_df_for_diff_datasets, models_metric_df])
        print(f'Extracted data for {dataset_name}')

    return models_metric_df_for_diff_datasets


def get_data_for_box_plots_for_diff_imputers_and_models(dataset_name: str, evaluation_scenario: str,
                                                        metric_name: str, group: str, db_client):
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

    # Add a baseline median to models_metric_df to display it as a horizontal line
    models_metric_df['Baseline_Median'] = None
    for model_name in models_metric_df['Model_Name'].unique():
        baseline_median = get_baseline_model_median(dataset_name=dataset_name,
                                                    model_name=model_name,
                                                    metric_name=metric_name,
                                                    db_client=db_client,
                                                    group=group)
        models_metric_df.loc[models_metric_df['Model_Name'] == model_name, 'Baseline_Median'] = baseline_median

    if metric_name == 'Accuracy':
        models_metric_df['Base_Rate'] = get_base_rate(dataset_name)

    return models_metric_df


def get_data_for_box_plots_for_diff_imputers_and_models_exp1(dataset_name: str, evaluation_scenario: str,
                                                             test_injection_scenario: str,
                                                             metric_name: str, group: str, db_client):
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
    
    # Add a baseline median to models_metric_df to display it as a horizontal line
    models_metric_df['Baseline_Median'] = None
    for model_name in models_metric_df['Model_Name'].unique():
        baseline_median = get_baseline_model_median(dataset_name=dataset_name,
                                                    model_name=model_name,
                                                    metric_name=metric_name,
                                                    db_client=db_client,
                                                    group=group)
        models_metric_df.loc[models_metric_df['Model_Name'] == model_name, 'Baseline_Median'] = baseline_median

    if metric_name == 'Accuracy':
        models_metric_df['Base_Rate'] = get_base_rate(dataset_name)

    return models_metric_df


def get_overall_metric_from_disparity_metric(disparity_metric):
    overall_to_disparity_metric_dct = {
        # Error disparity metrics
        TPR: [EQUALIZED_ODDS_TPR],
        TNR: [EQUALIZED_ODDS_TNR],
        FPR: [EQUALIZED_ODDS_FPR],
        FNR: [EQUALIZED_ODDS_FNR],
        ACCURACY: [ACCURACY_DIFFERENCE],
        SELECTION_RATE: [STATISTICAL_PARITY_DIFFERENCE, DISPARATE_IMPACT],
        # Stability disparity metrics
        LABEL_STABILITY: [LABEL_STABILITY_RATIO, LABEL_STABILITY_DIFFERENCE],
        JITTER: [JITTER_DIFFERENCE],
        IQR: [IQR_DIFFERENCE],
        # Uncertainty disparity metrics
        STD: [STD_DIFFERENCE, STD_RATIO],
        OVERALL_UNCERTAINTY: [OVERALL_UNCERTAINTY_DIFFERENCE, OVERALL_UNCERTAINTY_RATIO],
        ALEATORIC_UNCERTAINTY: [ALEATORIC_UNCERTAINTY_DIFFERENCE, ALEATORIC_UNCERTAINTY_RATIO],
        EPISTEMIC_UNCERTAINTY: [EPISTEMIC_UNCERTAINTY_DIFFERENCE, EPISTEMIC_UNCERTAINTY_RATIO],
    }
    for overall_metric in overall_to_disparity_metric_dct.keys():
        if disparity_metric in overall_to_disparity_metric_dct[overall_metric]:
            return overall_metric


def get_baseline_models_metric_df(db_client, dataset_name: str, metric_name: str, group: str):
    query = {
        'dataset_name': dataset_name,
        'null_imputer_name': 'baseline',
        'subgroup': group,
        'metric': metric_name,
        'tag': 'OK',
    }
    if db_client.db_name == 'data_cleaning_stability_3':
        metric_df = DB_CLIENT_2.read_metric_df_from_db(collection_name=EXP_COLLECTION_NAME,
                                                       query=query)
    else:
        metric_df = db_client.read_metric_df_from_db(collection_name=EXP_COLLECTION_NAME,
                                                     query=query)

    # Check uniqueness
    duplicates_mask = metric_df.duplicated(subset=['Exp_Pipeline_Guid', 'Model_Name', 'Subgroup', 'Metric'], keep=False)
    assert len(metric_df[duplicates_mask]) == 0, 'Metric df contains duplicates'

    columns_subset = ['Dataset_Name', 'Null_Imputer_Name', 'Virny_Random_State',
                      'Model_Name', 'Subgroup', 'Metric', 'Metric_Value']
    metric_df = metric_df[columns_subset]

    return metric_df


def get_baseline_model_metrics(dataset_name: str, model_name: str, metric_name: str,
                               db_client, group: str = 'overall'):
    if group == 'overall':
        models_metric_df = get_baseline_models_metric_df(db_client=db_client,
                                                         dataset_name=dataset_name,
                                                         metric_name=metric_name,
                                                         group=group)
    else:
        overall_metric = get_overall_metric_from_disparity_metric(disparity_metric=metric_name)
        models_metric_df = get_baseline_models_disparity_metric_df(db_client=db_client,
                                                                   dataset_name=dataset_name,
                                                                   metric_name=overall_metric,
                                                                   group=group)
        models_metric_df = models_metric_df[models_metric_df['Metric'] == metric_name]
        models_metric_df = models_metric_df.rename(columns={group: 'Metric_Value'})

    models_metric_df = models_metric_df[models_metric_df['Model_Name'] == model_name]
    return models_metric_df


def get_baseline_model_median(dataset_name: str, model_name: str, metric_name: str,
                              db_client, group: str = 'overall'):
    models_metric_df = get_baseline_model_metrics(dataset_name=dataset_name,
                                                  model_name=model_name,
                                                  metric_name=metric_name,
                                                  db_client=db_client,
                                                  group=group)
    return models_metric_df['Metric_Value'].median()


def get_models_metric_df(db_client, dataset_name: str, evaluation_scenario: str,
                         metric_name: str, group: str):
    query = {
        'dataset_name': dataset_name,
        'evaluation_scenario': evaluation_scenario,
        'metric': metric_name,
        'subgroup': group,
        'tag': 'OK',
    }
    metric_df = db_client.read_metric_df_from_db(collection_name=EXP_COLLECTION_NAME,
                                                 query=query)
    if db_client.db_name == 'data_cleaning_stability_2':
        metric_df2 = DB_CLIENT_2.read_metric_df_from_db(collection_name=EXP_COLLECTION_NAME,
                                                        query=query)
        metric_df = pd.concat([metric_df, metric_df2])

    # Check uniqueness
    duplicates_mask = metric_df.duplicated(subset=['Exp_Pipeline_Guid', 'Model_Name', 'Subgroup', 'Metric', 'Test_Set_Index'], keep=False)
    assert len(metric_df[duplicates_mask]) == 0, 'Metric df contains duplicates'

    columns_subset = ['Dataset_Name', 'Null_Imputer_Name', 'Evaluation_Scenario', 'Virny_Random_State',
                      'Model_Name', 'Subgroup', 'Metric', 'Metric_Value', 'Test_Set_Index']
    metric_df = metric_df[columns_subset]

    return metric_df


def get_baseline_models_disparity_metric_df(db_client, dataset_name: str, metric_name: str, group: str):
    dis_grp_models_metric_df = get_baseline_models_metric_df(db_client=db_client,
                                                             dataset_name=dataset_name,
                                                             metric_name=metric_name,
                                                             group=group + '_dis')
    priv_grp_models_metric_df = get_baseline_models_metric_df(db_client=db_client,
                                                              dataset_name=dataset_name,
                                                              metric_name=metric_name,
                                                              group=group + '_priv')
    grp_models_metric_df = pd.concat([dis_grp_models_metric_df, priv_grp_models_metric_df])

    # Compose group metrics
    disparity_metric_df = pd.DataFrame()
    for null_imputer_name in grp_models_metric_df['Null_Imputer_Name'].unique():
        for exp_seed in grp_models_metric_df['Virny_Random_State'].unique():
            for model_name in grp_models_metric_df['Model_Name'].unique():
                model_subgroup_metrics_df = grp_models_metric_df[
                    (grp_models_metric_df['Null_Imputer_Name'] == null_imputer_name) &
                    (grp_models_metric_df['Virny_Random_State'] == exp_seed) &
                    (grp_models_metric_df['Model_Name'] == model_name)
                    ]

                # Create columns based on values in the Subgroup column
                pivoted_model_subgroup_metrics_df = model_subgroup_metrics_df.pivot(columns='Subgroup', values='Metric_Value',
                                                                                    index=[col for col in model_subgroup_metrics_df.columns
                                                                                           if col not in ('Subgroup', 'Metric_Value')]).reset_index()
                pivoted_model_subgroup_metrics_df = pivoted_model_subgroup_metrics_df.rename_axis(None, axis=1)

                metrics_composer = MetricsComposer(
                    {model_name: pivoted_model_subgroup_metrics_df},
                    sensitive_attributes_dct={group: None}
                )
                model_group_metrics_df = metrics_composer.compose_metrics()
                model_group_metrics_df['Null_Imputer_Name'] = null_imputer_name
                model_group_metrics_df['Virny_Random_State'] = exp_seed
                model_group_metrics_df['Model_Name'] = model_name

                disparity_metric_df = pd.concat([disparity_metric_df, model_group_metrics_df])

    return disparity_metric_df


def get_models_disparity_metric_df(db_client, dataset_name: str, evaluation_scenario: str, metric_name: str, group: str):
    dis_grp_models_metric_df = get_models_metric_df(db_client=db_client,
                                                    dataset_name=dataset_name,
                                                    evaluation_scenario=evaluation_scenario,
                                                    metric_name=metric_name,
                                                    group=group + '_dis')
    priv_grp_models_metric_df = get_models_metric_df(db_client=db_client,
                                                     dataset_name=dataset_name,
                                                     evaluation_scenario=evaluation_scenario,
                                                     metric_name=metric_name,
                                                     group=group + '_priv')
    grp_models_metric_df = pd.concat([dis_grp_models_metric_df, priv_grp_models_metric_df])

    # Compose group metrics
    disparity_metric_df = pd.DataFrame()
    for null_imputer_name in grp_models_metric_df['Null_Imputer_Name'].unique():
        for exp_seed in grp_models_metric_df['Virny_Random_State'].unique():
            for model_name in grp_models_metric_df['Model_Name'].unique():
                for test_set_index in grp_models_metric_df['Test_Set_Index'].unique():
                    model_subgroup_metrics_df = grp_models_metric_df[
                        (grp_models_metric_df['Null_Imputer_Name'] == null_imputer_name) &
                        (grp_models_metric_df['Virny_Random_State'] == exp_seed) &
                        (grp_models_metric_df['Model_Name'] == model_name) &
                        (grp_models_metric_df['Test_Set_Index'] == test_set_index)
                        ]

                    # Create columns based on values in the Subgroup column
                    pivoted_model_subgroup_metrics_df = model_subgroup_metrics_df.pivot(columns='Subgroup', values='Metric_Value',
                                                                                        index=[col for col in model_subgroup_metrics_df.columns
                                                                                               if col not in ('Subgroup', 'Metric_Value')]).reset_index()
                    pivoted_model_subgroup_metrics_df = pivoted_model_subgroup_metrics_df.rename_axis(None, axis=1)

                    metrics_composer = MetricsComposer(
                        {model_name: pivoted_model_subgroup_metrics_df},
                        sensitive_attributes_dct={group: None}
                    )
                    model_group_metrics_df = metrics_composer.compose_metrics()
                    model_group_metrics_df['Null_Imputer_Name'] = null_imputer_name
                    model_group_metrics_df['Evaluation_Scenario'] = evaluation_scenario
                    model_group_metrics_df['Virny_Random_State'] = exp_seed
                    model_group_metrics_df['Model_Name'] = model_name
                    model_group_metrics_df['Test_Set_Index'] = test_set_index

                    disparity_metric_df = pd.concat([disparity_metric_df, model_group_metrics_df])

    return disparity_metric_df
