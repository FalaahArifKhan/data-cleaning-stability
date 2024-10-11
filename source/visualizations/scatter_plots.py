import altair as alt
import seaborn as sns
import pandas as pd
from altair.utils.schemapi import Undefined

from configs.constants import (ErrorRepairMethod, DIABETES_DATASET, GERMAN_CREDIT_DATASET, BANK_MARKETING_DATASET,
                               CARDIOVASCULAR_DISEASE_DATASET, ACS_INCOME_DATASET,
                               LAW_SCHOOL_DATASET, ACS_EMPLOYMENT_DATASET)
from source.visualizations.imputers_viz import get_data_for_box_plots_for_diff_imputers_and_datasets_for_mixed_exp
from source.visualizations.models_viz import get_data_for_box_plots_for_diff_imputers_and_datasets


def get_scatter_plot_data(missingness_types: list, dataset_to_column_name: dict, imputation_quality_metric_name: str,
                          extended_imputation_quality_metric_name: str, model_performance_metric_name: str,
                          db_client_1, db_client_3, dataset_to_group: dict = None, without_dummy: bool = False):
    imputation_quality_metrics_df = pd.DataFrame()
    model_performance_metrics_df = pd.DataFrame()
    for missingness_type in missingness_types:
        train_injection_scenario, test_injection_scenario = missingness_type['train'], missingness_type['test']
        if train_injection_scenario != 'mixed_exp':
            train_injection_scenario = train_injection_scenario.upper()
        test_injection_scenario = test_injection_scenario.upper()

        db_client = db_client_3 if train_injection_scenario == 'mixed_exp' else db_client_1
        imputation_quality_metrics_sub_df, new_imputation_quality_metric = (
            get_data_for_box_plots_for_diff_imputers_and_datasets_for_mixed_exp(train_injection_scenario=train_injection_scenario,
                                                                                test_injection_scenario=test_injection_scenario,
                                                                                metric_name=imputation_quality_metric_name,
                                                                                dataset_to_column_name=dataset_to_column_name,
                                                                                db_client=db_client,
                                                                                dataset_to_group=dataset_to_group,
                                                                                without_dummy=without_dummy))

        model_performance_metrics_sub_df = get_data_for_box_plots_for_diff_imputers_and_datasets(train_injection_scenario=train_injection_scenario,
                                                                                                 test_injection_scenario=test_injection_scenario,
                                                                                                 metric_name=model_performance_metric_name,
                                                                                                 db_client=db_client,
                                                                                                 dataset_to_group=dataset_to_group)

        imputation_quality_metrics_sub_df['Missingness_Type'] = train_injection_scenario + ' - ' + test_injection_scenario
        model_performance_metrics_sub_df['Missingness_Type'] = train_injection_scenario + ' - ' + test_injection_scenario
        imputation_quality_metrics_df = pd.concat([imputation_quality_metrics_df, imputation_quality_metrics_sub_df])
        model_performance_metrics_df = pd.concat([model_performance_metrics_df, model_performance_metrics_sub_df])

        print(f'Extraction for {missingness_type} is completed\n\n')

    if dataset_to_group is not None and new_imputation_quality_metric.lower() == 'rmse_difference':
        imputation_quality_metrics_df = imputation_quality_metrics_df.loc[
            ~((imputation_quality_metrics_df['Dataset_Name'] == GERMAN_CREDIT_DATASET) & (imputation_quality_metrics_df[new_imputation_quality_metric] > 100))
        ]

    model_performance_metrics_df = model_performance_metrics_df.rename(columns={'Metric_Value': model_performance_metric_name})
    avg_imputation_quality_metrics_df = imputation_quality_metrics_df.groupby(['Dataset_Name', 'Null_Imputer_Name', 'Evaluation_Scenario', 'Missingness_Type']).mean(numeric_only=True).reset_index()
    avg_model_performance_metrics_df = model_performance_metrics_df.groupby(['Dataset_Name', 'Null_Imputer_Name', 'Model_Name', 'Evaluation_Scenario', 'Missingness_Type']).mean(numeric_only=True).reset_index()

    # Merge avg_imputation_quality_metrics_df and avg_model_performance_metrics_df
    merged_df_1 = pd.merge(avg_imputation_quality_metrics_df, avg_model_performance_metrics_df[['Dataset_Name', 'Null_Imputer_Name', 'Model_Name', 'Evaluation_Scenario', 'Missingness_Type', model_performance_metric_name]],
                           on=['Dataset_Name', 'Null_Imputer_Name', 'Evaluation_Scenario', 'Missingness_Type'],
                           how='left')

    # Group by Missingness_Type and calculate min and max for both metrics
    min_df = merged_df_1.groupby('Missingness_Type').agg({
        model_performance_metric_name: ['min'],
        extended_imputation_quality_metric_name: ['min']
    }).reset_index()

    # Renaming columns for clarity
    min_df.columns = ['Missingness_Type',
                      model_performance_metric_name + '_Line',
                      extended_imputation_quality_metric_name + '_Line']

    max_df = merged_df_1.groupby('Missingness_Type').agg({
        model_performance_metric_name: ['max'],
        extended_imputation_quality_metric_name: ['max']
    }).reset_index()

    # Renaming columns for clarity
    max_df.columns = ['Missingness_Type',
                      model_performance_metric_name + '_Line',
                      extended_imputation_quality_metric_name + '_Line']

    min_max_df = pd.concat([min_df, max_df])

    # Merge merged_df_1 and min_max_df
    merged_df_2 = pd.merge(merged_df_1, min_max_df,
                           on=['Missingness_Type'],
                           how='left')

    return merged_df_2, new_imputation_quality_metric


def create_scatter_plot(missingness_types: list, dataset_to_column_name: dict,
                        imputation_quality_metric_name: str, model_performance_metric_name: str, shape_by: str,
                        db_client_1, db_client_3, dataset_to_group: dict = None, base_font_size: int = 22,
                        without_dummy: bool = False, ylim=Undefined):
    sns.set_style("whitegrid")
    columns_order = [missingness_type['train'] + ' - ' + missingness_type['test'] for missingness_type in missingness_types]
    imputers_order = ['deletion', 'median-mode', 'median-dummy', 'miss_forest',
                      'k_means_clustering', 'datawig', 'automl']
    if without_dummy:
        imputers_order = [t for t in imputers_order if t != ErrorRepairMethod.median_dummy.value]

    imputation_quality_metric_name = '_'.join([c.capitalize() for c in imputation_quality_metric_name.split('_')])
    model_performance_metric_name = '_'.join([c.capitalize() for c in model_performance_metric_name.split('_')]) \
        if 'equalized_odds' not in model_performance_metric_name.lower() else model_performance_metric_name
    extended_imputation_quality_metric_name = imputation_quality_metric_name
    imputation_quality_metric_name = imputation_quality_metric_name.lower().replace('_difference', '')

    # Read the data
    merged_df, new_imputation_quality_metric = get_scatter_plot_data(missingness_types=missingness_types,
                                                                     dataset_to_column_name=dataset_to_column_name,
                                                                     imputation_quality_metric_name=imputation_quality_metric_name,
                                                                     extended_imputation_quality_metric_name=extended_imputation_quality_metric_name,
                                                                     model_performance_metric_name=model_performance_metric_name,
                                                                     db_client_1=db_client_1,
                                                                     db_client_3=db_client_3,
                                                                     dataset_to_group=dataset_to_group,
                                                                     without_dummy=without_dummy)

    imputation_metric_title = new_imputation_quality_metric.replace('_', ' ')
    imputation_metric_title = (
        imputation_metric_title.replace('Rmse', 'RMSE')
        .replace('Kl Divergence Pred', 'KL Divergence Pred')
        .replace('Kl Divergence Total', 'KL Divergence Total')
        .replace('KL Divergence Pred Difference', 'KLD Pred Difference')
        .replace('KL Divergence Total Difference', 'KLD Total Difference')
    )
    model_performance_metric_title = model_performance_metric_name.replace('Equalized_Odds_', '') + 'D' if 'equalized_odds' in model_performance_metric_name.lower() else model_performance_metric_name.replace('_', ' ')

    # Create the scatter plot
    scatter_plot = (
        alt.Chart().mark_point(size=100).encode(
            x=alt.X(f'{model_performance_metric_name}:Q',
                    axis=alt.Axis(title=model_performance_metric_title)),
            y=alt.Y(f'{extended_imputation_quality_metric_name}:Q',
                    axis=alt.Axis(title=imputation_metric_title)),
            color=alt.Color("Null_Imputer_Name:N", title=None, sort=imputers_order),
            shape=alt.Shape(f"{shape_by}:N", title=None),
            # column=alt.Column('Missingness_Type:N', title=None, sort=columns_order)
        )
    )

    # Add dynamic y = x line (dashed)
    line = (
        alt.Chart().mark_line(
            strokeDash=[5, 5], color='gray'
        ).encode(
            x=f'{model_performance_metric_name}_Line:Q',
            y=f'{extended_imputation_quality_metric_name}_Line:Q',
            # column=alt.Column('Missingness_Type:N', title=None, sort=columns_order)
        )
    )

    # Add faceting based on Missingness_Type column
    faceted_plot = (
        alt.layer(
            scatter_plot, line,
            data=merged_df,
        ).properties(
            width=400,
            height=400,
        ).facet(
            column=alt.Column('Missingness_Type:N', title=None, sort=columns_order),
        )
    )

    final_chart = (
        faceted_plot.configure_legend(
            titleFontSize=base_font_size + 4,
            labelFontSize=base_font_size + 2,
            symbolStrokeWidth=10,
            labelLimit=400,
            titleLimit=300,
            columns=2,
            orient='top',
            direction='horizontal',
            titleAnchor='middle',
            symbolOffset=130 if dataset_to_group is not None else 100,
        ).configure_view(
            stroke=None
        ).configure_facet(
            spacing=30,
        ).configure_header(
            labelOrient='top',
            labelPadding=5,
            labelFontWeight='bold',
            labelFontSize=base_font_size + 6,
            titleFontSize=base_font_size + 6,
        ).configure_axis(
            labelFontSize=base_font_size + 4,
            titleFontSize=base_font_size + 6,
            labelFontWeight='normal',
            titleFontWeight='normal',
        ).configure_title(
            fontSize=base_font_size + 6,
        )
    )

    # Set a shared scale for the y-axis
    final_chart = final_chart.resolve_scale(y='independent')

    return final_chart
