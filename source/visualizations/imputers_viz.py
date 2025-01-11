import pandas as pd
import altair as alt
import seaborn as sns
from textwrap import wrap
from altair.utils.schemapi import Undefined

from configs.constants import (IMPUTATION_PERFORMANCE_METRICS_COLLECTION_NAME, ErrorRepairMethod,
                               DIABETES_DATASET, GERMAN_CREDIT_DATASET, BANK_MARKETING_DATASET,
                               CARDIOVASCULAR_DISEASE_DATASET, ACS_INCOME_DATASET,
                               LAW_SCHOOL_DATASET, ACS_EMPLOYMENT_DATASET)
from configs.datasets_config import DATASET_CONFIG
from configs.scenarios_config import EVALUATION_SCENARIOS_CONFIG
from source.visualizations.model_metrics_extraction_for_viz import get_evaluation_scenario
from source.custom_classes.database_client import DatabaseClient


DB_CLIENT_2 = DatabaseClient()
DB_CLIENT_2.connect()


def get_imputers_metric_df(db_client, dataset_name: str, evaluation_scenario: str,
                           column_name: str, group: str):
    query = {
        'dataset_name': dataset_name,
        'evaluation_scenario': evaluation_scenario,
        'column_with_nulls': column_name,
        'subgroup': group,
        'tag': 'OK',
    }
    metric_df = db_client.read_metric_df_from_db(collection_name=IMPUTATION_PERFORMANCE_METRICS_COLLECTION_NAME,
                                                 query=query)
    # if db_client.db_name == 'data_cleaning_stability_3':
    if db_client.db_name == 'data_cleaning_stability_2':
        metric_df2 = DB_CLIENT_2.read_metric_df_from_db(collection_name=IMPUTATION_PERFORMANCE_METRICS_COLLECTION_NAME,
                                                        query=query)
        metric_df = pd.concat([metric_df, metric_df2])

    # Check uniqueness
    duplicates_mask = metric_df.duplicated(subset=['Imputation_Guid'], keep=False)
    assert len(metric_df[duplicates_mask]) == 0, 'Metric df contains duplicates'

    return metric_df


def get_imputers_disparity_metric_df(db_client, dataset_name: str, evaluation_scenario: str,
                                     column_name: str, metric_name: str, group: str):
    dis_grp_imputers_metric_df = get_imputers_metric_df(db_client=db_client,
                                                        dataset_name=dataset_name,
                                                        evaluation_scenario=evaluation_scenario,
                                                        column_name=column_name,
                                                        group=group + '_dis')
    priv_grp_imputers_metric_df = get_imputers_metric_df(db_client=db_client,
                                                         dataset_name=dataset_name,
                                                         evaluation_scenario=evaluation_scenario,
                                                         column_name=column_name,
                                                         group=group + '_priv')

    columns_subset = ['Dataset_Name', 'Null_Imputer_Name', 'Evaluation_Scenario',
                      'Experiment_Seed', 'Dataset_Part', 'Column_With_Nulls', metric_name]
    dis_grp_imputers_metric_df = dis_grp_imputers_metric_df[columns_subset]
    priv_grp_imputers_metric_df = priv_grp_imputers_metric_df[columns_subset]

    merged_imputers_metric_df = pd.merge(dis_grp_imputers_metric_df, priv_grp_imputers_metric_df,
                                         on=['Dataset_Name', 'Null_Imputer_Name', 'Evaluation_Scenario',
                                             'Experiment_Seed', 'Dataset_Part', 'Column_With_Nulls'],
                                         how='left',
                                         suffixes=('_dis', '_priv'))
    merged_imputers_metric_df[metric_name + '_Difference'] = \
            merged_imputers_metric_df[metric_name + '_dis'] - merged_imputers_metric_df[metric_name + '_priv']

    return merged_imputers_metric_df


def create_box_plots_for_diff_imputers_and_single_eval_scenario(dataset_name: str, evaluation_scenario: str,
                                                                column_name: str, metric_name: str,
                                                                db_client, title: str,
                                                                group: str = 'overall', base_font_size: int = 18,
                                                                without_dummy: bool = False, ylim=Undefined):
    sns.set_style("whitegrid")
    imputers_order = ['deletion', 'median-mode', 'median-dummy', 'miss_forest',
                      'k_means_clustering', 'datawig', 'automl']
    if without_dummy:
        imputers_order = [t for t in imputers_order if t != ErrorRepairMethod.median_dummy.value]

    metric_name = '_'.join([c.capitalize() for c in metric_name.split('_')])
    if group == 'overall':
        imputers_metric_df = get_imputers_metric_df(db_client=db_client,
                                                    dataset_name=dataset_name,
                                                    evaluation_scenario=evaluation_scenario,
                                                    column_name=column_name,
                                                    group=group)
    else:
        imputers_metric_df = get_imputers_disparity_metric_df(db_client=db_client,
                                                              dataset_name=dataset_name,
                                                              evaluation_scenario=evaluation_scenario,
                                                              column_name=column_name,
                                                              metric_name=metric_name,
                                                              group=group)
        metric_name = metric_name + '_Difference'

    if without_dummy:
        to_plot = imputers_metric_df[
                    (imputers_metric_df['Dataset_Part'].str.contains('X_test')) &
                    (imputers_metric_df['Null_Imputer_Name'] != ErrorRepairMethod.median_dummy.value)
                  ]
    else:
        to_plot = imputers_metric_df[imputers_metric_df['Dataset_Part'].str.contains('X_test')]

    to_plot['Test_Injection_Strategy'] = to_plot['Dataset_Part'].apply(lambda x: x.split('_')[-1][:-1])

    metric_title = metric_name.replace('_', ' ')
    metric_title = (
        metric_title.replace('Rmse', 'RMSE')
            .replace('Kl Divergence Pred', 'KL Divergence Pred')
            .replace('Kl Divergence Total', 'KL Divergence Total')
    )

    chart = (
        alt.Chart(to_plot).mark_boxplot(
            ticks=True,
            median={'stroke': 'black', 'strokeWidth': 0.7},
        ).encode(
            x=alt.X("Null_Imputer_Name:N",
                    title=None,
                    sort=imputers_order,
                    axis=alt.Axis(labels=False)),
            y=alt.Y(f"{metric_name}:Q",
                    title=metric_title,
                    scale=alt.Scale(zero=False, domain=ylim)),
            color=alt.Color("Null_Imputer_Name:N", title=None, sort=imputers_order),
            column=alt.Column('Test_Injection_Strategy:N',
                              title=None,
                              sort=['MCAR', 'MAR', 'MNAR'])
        ).properties(
            width=120,
            title=alt.TitleParams(text=title, fontSize=base_font_size + 6, anchor='middle', align='center', dx=40),
        )
    )

    return chart


def create_box_plots_for_diff_imputers(dataset_name: str, column_name: str,
                                       metric_name: str, db_client,
                                       group: str = 'overall', without_dummy: bool = False,
                                       ylim=Undefined):
    base_font_size = 20
    base_chart1 = create_box_plots_for_diff_imputers_and_single_eval_scenario(dataset_name=dataset_name,
                                                                              evaluation_scenario='exp1_mcar3',
                                                                              title='MCAR train set',
                                                                              column_name=column_name,
                                                                              metric_name=metric_name,
                                                                              db_client=db_client,
                                                                              group=group,
                                                                              base_font_size=base_font_size,
                                                                              without_dummy=without_dummy,
                                                                              ylim=ylim)
    base_chart2 = create_box_plots_for_diff_imputers_and_single_eval_scenario(dataset_name=dataset_name,
                                                                              evaluation_scenario='exp1_mar3',
                                                                              title='MAR train set',
                                                                              column_name=column_name,
                                                                              metric_name=metric_name,
                                                                              db_client=db_client,
                                                                              group=group,
                                                                              base_font_size=base_font_size,
                                                                              without_dummy=without_dummy,
                                                                              ylim=ylim)
    base_chart3 = create_box_plots_for_diff_imputers_and_single_eval_scenario(dataset_name=dataset_name,
                                                                              evaluation_scenario='exp1_mnar3',
                                                                              title='MNAR train set',
                                                                              column_name=column_name,
                                                                              metric_name=metric_name,
                                                                              db_client=db_client,
                                                                              group=group,
                                                                              base_font_size=base_font_size,
                                                                              without_dummy=without_dummy,
                                                                              ylim=ylim)

    # Concatenate two base charts
    main_base_chart = alt.hconcat()
    main_base_chart |= base_chart1
    main_base_chart |= base_chart2
    main_base_chart |= base_chart3

    final_grid_chart = (
        main_base_chart.configure_axis(
            labelFontSize=base_font_size + 4,
            titleFontSize=base_font_size + 6,
            labelFontWeight='normal',
            titleFontWeight='normal',
        ).configure_title(
            fontSize=base_font_size + 2
        ).configure_legend(
            titleFontSize=base_font_size + 4,
            labelFontSize=base_font_size + 2,
            symbolStrokeWidth=10,
            labelLimit=400,
            titleLimit=300,
            columns=4,
            orient='top',
            direction='horizontal',
            titleAnchor='middle',
            symbolOffset=150 if without_dummy else 120,
        ).configure_facet(
            spacing=10
        ).configure_view(
            stroke=None
        ).configure_header(
            labelOrient='bottom',
            labelPadding=5,
            labelFontSize=base_font_size + 2,
            titleFontSize=base_font_size + 2,
        ).configure_axis(
            labelFontSize=base_font_size + 4,
            titleFontSize=base_font_size + 6,
            labelFontWeight='normal',
            titleFontWeight='normal',
        )
    )

    # Set a shared scale for the y-axis
    final_grid_chart = final_grid_chart.resolve_scale(y='shared')

    return final_grid_chart


def create_box_plots_for_mixed_exp(dataset_name: str, column_names: list, metric_name: str, db_client,
                                   group: str = 'overall', base_font_size: int = 18,
                                   without_dummy: bool = False, ylim=Undefined):
    evaluation_scenario = 'mixed_exp'

    sns.set_style("whitegrid")
    imputers_order = ['deletion', 'median-mode', 'median-dummy', 'miss_forest',
                      'k_means_clustering', 'datawig', 'automl']
    if without_dummy:
        imputers_order = [t for t in imputers_order if t != ErrorRepairMethod.median_dummy.value]

    metric_name = '_'.join([c.capitalize() for c in metric_name.split('_')])
    imputers_metric_df = pd.DataFrame()
    for column_name in column_names:
        if group == 'overall':
            column_imputers_metric_df = get_imputers_metric_df(db_client=db_client,
                                                               dataset_name=dataset_name,
                                                               evaluation_scenario=evaluation_scenario,
                                                               column_name=column_name,
                                                               group=group)
        else:
            column_imputers_metric_df = get_imputers_disparity_metric_df(db_client=db_client,
                                                                         dataset_name=dataset_name,
                                                                         evaluation_scenario=evaluation_scenario,
                                                                         column_name=column_name,
                                                                         metric_name=metric_name,
                                                                         group=group)

        imputers_metric_df = pd.concat([imputers_metric_df, column_imputers_metric_df])

    if group != 'overall':
        metric_name = metric_name + '_Difference'

    if without_dummy:
        to_plot = imputers_metric_df[
            (imputers_metric_df['Dataset_Part'].str.contains('X_test')) &
            (imputers_metric_df['Null_Imputer_Name'] != ErrorRepairMethod.median_dummy.value)
            ]
    else:
        to_plot = imputers_metric_df[imputers_metric_df['Dataset_Part'].str.contains('X_test')]

    metric_title = metric_name.replace('_', ' ')
    metric_title = (
        metric_title.replace('Rmse', 'RMSE')
        .replace('Kl Divergence Pred', 'KL Divergence Pred')
        .replace('Kl Divergence Total', 'KL Divergence Total')
    )

    chart = (
        alt.Chart(to_plot).mark_boxplot(
            ticks=True,
            median={'stroke': 'black', 'strokeWidth': 0.7},
        ).encode(
            x=alt.X("Null_Imputer_Name:N",
                    title=None,
                    sort=imputers_order,
                    axis=alt.Axis(labels=False)),
            y=alt.Y(f"{metric_name}:Q",
                    title=metric_title,
                    scale=alt.Scale(zero=False, domain=ylim)),
            color=alt.Color("Null_Imputer_Name:N", title=None, sort=imputers_order),
            column=alt.Column('Column_With_Nulls:N', title=None)
        ).properties(
            width=180,
        )
    )

    final_grid_chart = (
        chart.configure_axis(
            labelFontSize=base_font_size + 4,
            titleFontSize=base_font_size + 6,
            labelFontWeight='normal',
            titleFontWeight='normal',
        ).configure_title(
            fontSize=base_font_size + 2
        ).configure_legend(
            titleFontSize=base_font_size + 4,
            labelFontSize=base_font_size + 2,
            symbolStrokeWidth=10,
            labelLimit=400,
            titleLimit=300,
            columns=4,
            orient='top',
            direction='horizontal',
            titleAnchor='middle',
            symbolOffset=10,
        ).configure_facet(
            spacing=10
        ).configure_view(
            stroke=None
        ).configure_header(
            labelOrient='bottom',
            labelPadding=5,
            labelFontSize=base_font_size + 2,
            titleFontSize=base_font_size + 2,
        ).configure_axis(
            labelFontSize=base_font_size + 4,
            titleFontSize=base_font_size + 6,
            labelFontWeight='normal',
            titleFontWeight='normal',
        )
    )

    return final_grid_chart


def get_line_bands_for_diff_imputers_and_single_test_set(models_metric_df, test_set: str, metric_name: str, title: str,
                                                         base_font_size: int = 18, ylim=Undefined, with_band=True,
                                                         without_dummy=False):
    imputers_order = ['deletion', 'median-mode', 'median-dummy', 'miss_forest',
                      'k_means_clustering', 'datawig', 'automl']
    if without_dummy:
        imputers_order = [t for t in imputers_order if t != ErrorRepairMethod.median_dummy.value]

    title = f'{title} & {test_set} test'
    models_metric_df_for_test_set = models_metric_df[models_metric_df['Test_Injection_Strategy'] == test_set]

    metric_title = metric_name.replace('_', ' ')
    metric_title = (
        metric_title.replace('Rmse', 'RMSE')
        .replace('Kl Divergence Pred', 'KL Divergence Pred')
        .replace('Kl Divergence Total', 'KL Divergence Total')
    )

    line_chart = alt.Chart(models_metric_df_for_test_set).mark_line().encode(
        x=alt.X(field='Test_Error_Rate', type='quantitative', title='Test Error Rate',
                scale=alt.Scale(nice=False), axis=alt.Axis(labelExpr=f"(datum.value == 0.1) || (datum.value == 0.3) || (datum.value == 0.5) ? datum.label : ''")),
        y=alt.Y(f'mean({metric_name})', type='quantitative', title=metric_title, scale=alt.Scale(zero=False, domain=ylim)),
        color=alt.Color('Null_Imputer_Name:N', title=None, sort=imputers_order),
    )
    if with_band:
        band_chart = alt.Chart(models_metric_df_for_test_set).mark_errorband(extent="stdev").encode(
            x=alt.X(field='Test_Error_Rate', type='quantitative', title='Test Error Rate',
                    scale=alt.Scale(nice=False), axis=alt.Axis(labelExpr=f"(datum.value == 0.1) || (datum.value == 0.3) || (datum.value == 0.5) ? datum.label : ''")),
            y=alt.Y(field=metric_name, type='quantitative', title=metric_title, scale=alt.Scale(zero=False, domain=ylim)),
            color=alt.Color('Null_Imputer_Name:N', title=None, sort=imputers_order),
        )
        base_chart = (band_chart + line_chart)
    else:
        base_chart = line_chart

    base_chart = base_chart.properties(
        width=200, height=200,
        title=alt.TitleParams(text=title, fontSize=base_font_size + 2),
    )

    return base_chart


def get_line_bands_for_diff_imputers_and_single_eval_scenario(dataset_name: str, evaluation_scenario: str,
                                                              column_name: str, metric_name: str, db_client,
                                                              title: str, group: str = 'overall',
                                                              base_font_size: int = 18, ylim=Undefined,
                                                              with_band=True, without_dummy=False):
    sns.set_style("whitegrid")
    metric_name = '_'.join([c.capitalize() for c in metric_name.split('_')])
    if group == 'overall':
        imputers_metric_df = get_imputers_metric_df(db_client=db_client,
                                                    dataset_name=dataset_name,
                                                    evaluation_scenario=evaluation_scenario,
                                                    column_name=column_name,
                                                    group=group)
    else:
        imputers_metric_df = get_imputers_disparity_metric_df(db_client=db_client,
                                                              dataset_name=dataset_name,
                                                              evaluation_scenario=evaluation_scenario,
                                                              column_name=column_name,
                                                              metric_name=metric_name,
                                                              group=group)
        metric_name = metric_name + '_Difference'

    if without_dummy:
        to_plot = imputers_metric_df[
            (imputers_metric_df['Dataset_Part'].str.contains('X_test')) &
            (imputers_metric_df['Null_Imputer_Name'] != ErrorRepairMethod.median_dummy.value)
            ]
    else:
        to_plot = imputers_metric_df[imputers_metric_df['Dataset_Part'].str.contains('X_test')]

    to_plot['Test_Injection_Strategy'] = to_plot['Dataset_Part'].apply(lambda x: x.split('_')[-1][:-1])
    to_plot['Test_Error_Rate'] = to_plot['Dataset_Part'].apply(lambda x: 0.1 * int(x.split('_')[-1][-1]))

    mcar_base_chart = get_line_bands_for_diff_imputers_and_single_test_set(models_metric_df=to_plot,
                                                                           test_set='MCAR',
                                                                           metric_name=metric_name,
                                                                           title=title,
                                                                           base_font_size=base_font_size,
                                                                           ylim=ylim,
                                                                           with_band=with_band,
                                                                           without_dummy=without_dummy)

    mar_base_chart = get_line_bands_for_diff_imputers_and_single_test_set(models_metric_df=to_plot,
                                                                          test_set='MAR',
                                                                          metric_name=metric_name,
                                                                          title=title,
                                                                          base_font_size=base_font_size,
                                                                          ylim=ylim,
                                                                          with_band=with_band,
                                                                          without_dummy=without_dummy)

    mnar_base_chart = get_line_bands_for_diff_imputers_and_single_test_set(models_metric_df=to_plot,
                                                                           test_set='MNAR',
                                                                           metric_name=metric_name,
                                                                           title=title,
                                                                           base_font_size=base_font_size,
                                                                           ylim=ylim,
                                                                           with_band=with_band,
                                                                           without_dummy=without_dummy)

    return mcar_base_chart, mar_base_chart, mnar_base_chart


def create_line_bands_for_diff_imputers(dataset_name: str, column_name: str, metric_name: str, db_client,
                                        group: str = 'overall', ylim=Undefined, with_band=True,
                                        without_dummy: bool = False):
    base_font_size = 20
    mcar_base_chart1, mar_base_chart1, mnar_base_chart1 = (
        get_line_bands_for_diff_imputers_and_single_eval_scenario(dataset_name=dataset_name,
                                                                  evaluation_scenario='exp2_3_mcar3',
                                                                  title='MCAR train',
                                                                  column_name=column_name,
                                                                  metric_name=metric_name,
                                                                  db_client=db_client,
                                                                  group=group,
                                                                  base_font_size=base_font_size,
                                                                  ylim=ylim,
                                                                  with_band=with_band,
                                                                  without_dummy=without_dummy))
    print('Prepared a plot for an MCAR train set')
    mcar_base_chart2, mar_base_chart2, mnar_base_chart2 = (
        get_line_bands_for_diff_imputers_and_single_eval_scenario(dataset_name=dataset_name,
                                                                  evaluation_scenario='exp2_3_mar3',
                                                                  title='MAR train',
                                                                  column_name=column_name,
                                                                  metric_name=metric_name,
                                                                  db_client=db_client,
                                                                  group=group,
                                                                  base_font_size=base_font_size,
                                                                  ylim=ylim,
                                                                  with_band=with_band,
                                                                  without_dummy=without_dummy))
    print('Prepared a plot for an MAR train set')
    mcar_base_chart3, mar_base_chart3, mnar_base_chart3 = (
        get_line_bands_for_diff_imputers_and_single_eval_scenario(dataset_name=dataset_name,
                                                                  evaluation_scenario='exp2_3_mnar3',
                                                                  title='MNAR train',
                                                                  column_name=column_name,
                                                                  metric_name=metric_name,
                                                                  db_client=db_client,
                                                                  group=group,
                                                                  base_font_size=base_font_size,
                                                                  ylim=ylim,
                                                                  with_band=with_band,
                                                                  without_dummy=without_dummy))
    print('Prepared a plot for an MNAR train set')

    # Concatenate two base charts
    main_base_chart = alt.vconcat()

    row1 = alt.hconcat()
    row1 |= mcar_base_chart1
    row1 |= mar_base_chart1
    row1 |= mnar_base_chart1

    row2 = alt.hconcat()
    row2 |= mcar_base_chart2
    row2 |= mar_base_chart2
    row2 |= mnar_base_chart2

    row3 = alt.hconcat()
    row3 |= mcar_base_chart3
    row3 |= mar_base_chart3
    row3 |= mnar_base_chart3

    main_base_chart &= row1.resolve_scale(y='shared')
    main_base_chart &= row2.resolve_scale(y='shared')
    main_base_chart &= row3.resolve_scale(y='shared')

    final_grid_chart = (
        main_base_chart.configure_axis(
            labelFontSize=base_font_size + 2,
            titleFontSize=base_font_size + 2,
            labelFontWeight='normal',
            titleFontWeight='normal',
        ).configure_title(
            fontSize=base_font_size + 2
        ).configure_legend(
            titleFontSize=base_font_size + 2,
            labelFontSize=base_font_size + 2,
            symbolStrokeWidth=10,
            labelLimit=400,
            titleLimit=300,
            columns=3,
            orient='top',
            direction='horizontal',
            titleAnchor='middle',
            symbolOffset=80,
        )
    )

    # Set a shared scale for the y-axis
    final_grid_chart = final_grid_chart.resolve_scale(y='shared')

    return final_grid_chart


def create_line_bands_for_no_shift(dataset_name: str, column_name: str, metric_name: str, db_client,
                                   group: str = 'overall', ylim=Undefined, with_band=True,
                                   without_dummy: bool = False):
    base_font_size = 20
    mcar_base_chart1, _, _ = (
        get_line_bands_for_diff_imputers_and_single_eval_scenario(dataset_name=dataset_name,
                                                                  evaluation_scenario='exp2_3_mcar3',
                                                                  title='MCAR train',
                                                                  column_name=column_name,
                                                                  metric_name=metric_name,
                                                                  db_client=db_client,
                                                                  group=group,
                                                                  base_font_size=base_font_size,
                                                                  ylim=ylim,
                                                                  with_band=with_band,
                                                                  without_dummy=without_dummy))
    print('Prepared a plot for an MCAR train set')
    _, mar_base_chart2, _ = (
        get_line_bands_for_diff_imputers_and_single_eval_scenario(dataset_name=dataset_name,
                                                                  evaluation_scenario='exp2_3_mar3',
                                                                  title='MAR train',
                                                                  column_name=column_name,
                                                                  metric_name=metric_name,
                                                                  db_client=db_client,
                                                                  group=group,
                                                                  base_font_size=base_font_size,
                                                                  ylim=ylim,
                                                                  with_band=with_band,
                                                                  without_dummy=without_dummy))
    print('Prepared a plot for an MAR train set')
    _, _, mnar_base_chart3 = (
        get_line_bands_for_diff_imputers_and_single_eval_scenario(dataset_name=dataset_name,
                                                                  evaluation_scenario='exp2_3_mnar3',
                                                                  title='MNAR train',
                                                                  column_name=column_name,
                                                                  metric_name=metric_name,
                                                                  db_client=db_client,
                                                                  group=group,
                                                                  base_font_size=base_font_size,
                                                                  ylim=ylim,
                                                                  with_band=with_band,
                                                                  without_dummy=without_dummy))
    print('Prepared a plot for an MNAR train set')

    # Concatenate two base charts
    main_base_chart = alt.hconcat()
    main_base_chart |= mcar_base_chart1
    main_base_chart |= mar_base_chart2
    main_base_chart |= mnar_base_chart3

    final_grid_chart = (
        main_base_chart.configure_axis(
            labelFontSize=base_font_size + 2,
            titleFontSize=base_font_size + 2,
            labelFontWeight='normal',
            titleFontWeight='normal',
        ).configure_title(
            fontSize=base_font_size + 2
        ).configure_legend(
            titleFontSize=base_font_size + 2,
            labelFontSize=base_font_size + 2,
            symbolStrokeWidth=10,
            labelLimit=400,
            titleLimit=300,
            columns=3,
            orient='top',
            direction='horizontal',
            titleAnchor='middle',
            symbolOffset=80,
        )
    )

    # Set a shared scale for the y-axis
    final_grid_chart = final_grid_chart.resolve_scale(y='shared')

    return final_grid_chart


def get_exp2_line_bands_for_diff_imputers_and_single_test_set(models_metric_df, test_set: str, metric_name: str, train_set: str,
                                                              base_font_size: int = 18, ylim=Undefined, with_band=True,
                                                              without_dummy=False):
    imputers_order = ['deletion', 'median-mode', 'median-dummy', 'miss_forest',
                      'k_means_clustering', 'datawig', 'automl']
    if without_dummy:
        imputers_order = [t for t in imputers_order if t != ErrorRepairMethod.median_dummy.value]

    title = f'{train_set} train & {test_set} test'
    models_metric_df_for_test_set = models_metric_df[models_metric_df['Test_Injection_Strategy'] == test_set]

    metric_title = metric_name.replace('_', ' ')
    metric_title = (
        metric_title.replace('Rmse', 'RMSE')
        .replace('Kl Divergence Pred', 'KL Divergence Pred')
        .replace('Kl Divergence Total', 'KL Divergence Total')
    )

    line_chart = alt.Chart(models_metric_df_for_test_set).mark_line().encode(
        x=alt.X(field='Train_Error_Rate', type='quantitative', title='Train Error Rate',
                scale=alt.Scale(nice=False), axis=alt.Axis(labelExpr=f"(datum.value == 0.1) || (datum.value == 0.3) || (datum.value == 0.5) ? datum.label : ''")),
        y=alt.Y(f'mean({metric_name})', type='quantitative', title=metric_title, scale=alt.Scale(zero=False, domain=ylim)),
        color=alt.Color('Null_Imputer_Name:N', title=None, sort=imputers_order),
    )
    if with_band:
        band_chart = alt.Chart(models_metric_df_for_test_set).mark_errorband(extent="stdev").encode(
            x=alt.X(field='Train_Error_Rate', type='quantitative', title='Train Error Rate',
                    scale=alt.Scale(nice=False), axis=alt.Axis(labelExpr=f"(datum.value == 0.1) || (datum.value == 0.3) || (datum.value == 0.5) ? datum.label : ''")),
            y=alt.Y(field=metric_name, type='quantitative', title=metric_title, scale=alt.Scale(zero=False, domain=ylim)),
            color=alt.Color('Null_Imputer_Name:N', title=None, sort=imputers_order),
        )
        base_chart = (band_chart + line_chart)
    else:
        base_chart = line_chart

    base_chart = base_chart.properties(
        width=200, height=200,
        title=alt.TitleParams(text=title, fontSize=base_font_size + 2),
    )

    return base_chart


def get_exp2_line_bands_for_diff_imputers_and_single_eval_scenario(dataset_name: str, evaluation_scenarios: list,
                                                                   column_name: str, metric_name: str, db_client,
                                                                   train_set: str, group: str = 'overall',
                                                                   base_font_size: int = 18, ylim=Undefined,
                                                                   without_dummy=True, with_band=True):
    sns.set_style("whitegrid")
    metric_name = '_'.join([c.capitalize() for c in metric_name.split('_')])

    imputers_metric_df = pd.DataFrame()
    for evaluation_scenario in evaluation_scenarios:
        if group == 'overall':
            scenarios_imputers_metric_df = get_imputers_metric_df(db_client=db_client,
                                                                  dataset_name=dataset_name,
                                                                  evaluation_scenario=evaluation_scenario,
                                                                  column_name=column_name,
                                                                  group=group)
        else:
            scenarios_imputers_metric_df = get_imputers_disparity_metric_df(db_client=db_client,
                                                                            dataset_name=dataset_name,
                                                                            evaluation_scenario=evaluation_scenario,
                                                                            column_name=column_name,
                                                                            metric_name=metric_name,
                                                                            group=group)

        imputers_metric_df = pd.concat([imputers_metric_df, scenarios_imputers_metric_df])

    metric_name = metric_name if group == 'overall' else metric_name + '_Difference'
    if without_dummy:
        to_plot = imputers_metric_df[
            (imputers_metric_df['Dataset_Part'].str.contains('X_test')) &
            (imputers_metric_df['Null_Imputer_Name'] != ErrorRepairMethod.median_dummy.value)
            ]
    else:
        to_plot = imputers_metric_df[imputers_metric_df['Dataset_Part'].str.contains('X_test')]

    to_plot['Train_Injection_Strategy'] = to_plot.apply(
        lambda row: EVALUATION_SCENARIOS_CONFIG[row['Evaluation_Scenario']]['train_injection_scenario'][:-1],
        axis=1
    )
    to_plot['Train_Error_Rate'] = to_plot.apply(
        lambda row: 0.1 * int(EVALUATION_SCENARIOS_CONFIG[row['Evaluation_Scenario']]['train_injection_scenario'][-1]),
        axis=1
    )

    to_plot = to_plot[to_plot['Train_Injection_Strategy'] == train_set]
    to_plot = to_plot[to_plot['Dataset_Part'].isin(['X_test_MCAR3', 'X_test_MAR3', 'X_test_MNAR3'])]

    to_plot['Test_Injection_Strategy'] = to_plot['Dataset_Part'].apply(lambda x: x.split('_')[-1][:-1])
    to_plot['Test_Error_Rate'] = to_plot['Dataset_Part'].apply(lambda x: 0.1 * int(x.split('_')[-1][-1]))

    mcar_base_chart = get_exp2_line_bands_for_diff_imputers_and_single_test_set(models_metric_df=to_plot,
                                                                                test_set='MCAR',
                                                                                metric_name=metric_name,
                                                                                train_set=train_set,
                                                                                base_font_size=base_font_size,
                                                                                ylim=ylim,
                                                                                with_band=with_band,
                                                                                without_dummy=without_dummy)

    mar_base_chart = get_exp2_line_bands_for_diff_imputers_and_single_test_set(models_metric_df=to_plot,
                                                                               test_set='MAR',
                                                                               metric_name=metric_name,
                                                                               train_set=train_set,
                                                                               base_font_size=base_font_size,
                                                                               ylim=ylim,
                                                                               with_band=with_band,
                                                                               without_dummy=without_dummy)

    mnar_base_chart = get_exp2_line_bands_for_diff_imputers_and_single_test_set(models_metric_df=to_plot,
                                                                                test_set='MNAR',
                                                                                metric_name=metric_name,
                                                                                train_set=train_set,
                                                                                base_font_size=base_font_size,
                                                                                ylim=ylim,
                                                                                with_band=with_band,
                                                                                without_dummy=without_dummy)

    return mcar_base_chart, mar_base_chart, mnar_base_chart


def create_exp2_line_bands_for_diff_imputers(dataset_name: str, column_name: str, metric_name: str, db_client,
                                             group: str = 'overall', ylim=Undefined, with_band=True,
                                             without_dummy: bool = False):
    base_font_size = 20
    mcar_base_chart1, mar_base_chart1, mnar_base_chart1 = (
        get_exp2_line_bands_for_diff_imputers_and_single_eval_scenario(dataset_name=dataset_name,
                                                                       evaluation_scenarios=['exp2_3_mcar1', 'exp2_3_mcar3', 'exp2_3_mcar5'],
                                                                       train_set='MCAR',
                                                                       column_name=column_name,
                                                                       metric_name=metric_name,
                                                                       db_client=db_client,
                                                                       group=group,
                                                                       base_font_size=base_font_size,
                                                                       ylim=ylim,
                                                                       with_band=with_band,
                                                                       without_dummy=without_dummy))
    print('Prepared a plot for an MCAR train set')
    mcar_base_chart2, mar_base_chart2, mnar_base_chart2 = (
        get_exp2_line_bands_for_diff_imputers_and_single_eval_scenario(dataset_name=dataset_name,
                                                                       evaluation_scenarios=['exp2_3_mar1', 'exp2_3_mar3', 'exp2_3_mar5'],
                                                                       train_set='MAR',
                                                                       column_name=column_name,
                                                                       metric_name=metric_name,
                                                                       db_client=db_client,
                                                                       group=group,
                                                                       base_font_size=base_font_size,
                                                                       ylim=ylim,
                                                                       with_band=with_band,
                                                                       without_dummy=without_dummy))
    print('Prepared a plot for an MAR train set')
    mcar_base_chart3, mar_base_chart3, mnar_base_chart3 = (
        get_exp2_line_bands_for_diff_imputers_and_single_eval_scenario(dataset_name=dataset_name,
                                                                       evaluation_scenarios=['exp2_3_mnar1', 'exp2_3_mnar3', 'exp2_3_mnar5'],
                                                                       train_set='MNAR',
                                                                       column_name=column_name,
                                                                       metric_name=metric_name,
                                                                       db_client=db_client,
                                                                       group=group,
                                                                       base_font_size=base_font_size,
                                                                       ylim=ylim,
                                                                       with_band=with_band,
                                                                       without_dummy=without_dummy))
    print('Prepared a plot for an MNAR train set')

    # Concatenate two base charts
    main_base_chart = alt.vconcat()

    row1 = alt.hconcat()
    row1 |= mcar_base_chart1
    row1 |= mar_base_chart1
    row1 |= mnar_base_chart1

    row2 = alt.hconcat()
    row2 |= mcar_base_chart2
    row2 |= mar_base_chart2
    row2 |= mnar_base_chart2

    row3 = alt.hconcat()
    row3 |= mcar_base_chart3
    row3 |= mar_base_chart3
    row3 |= mnar_base_chart3

    main_base_chart &= row1.resolve_scale(y='shared')
    main_base_chart &= row2.resolve_scale(y='shared')
    main_base_chart &= row3.resolve_scale(y='shared')

    final_grid_chart = (
        main_base_chart.configure_axis(
            labelFontSize=base_font_size + 2,
            titleFontSize=base_font_size + 2,
            labelFontWeight='normal',
            titleFontWeight='normal',
        ).configure_title(
            fontSize=base_font_size + 2
        ).configure_legend(
            titleFontSize=base_font_size + 2,
            labelFontSize=base_font_size + 2,
            symbolStrokeWidth=10,
            labelLimit=400,
            titleLimit=300,
            columns=3,
            orient='top',
            direction='horizontal',
            titleAnchor='middle',
            symbolOffset=80,
        )
    )

    # Set a shared scale for the y-axis
    final_grid_chart = final_grid_chart.resolve_scale(y='shared')

    return final_grid_chart


def create_exp2_line_bands_for_no_shift(dataset_name: str, column_name: str, metric_name: str, db_client,
                                        group: str = 'overall', ylim=Undefined, with_band=True,
                                        without_dummy: bool = False):
    base_font_size = 20
    mcar_base_chart1, _, _ = (
        get_exp2_line_bands_for_diff_imputers_and_single_eval_scenario(dataset_name=dataset_name,
                                                                       evaluation_scenarios=['exp2_3_mcar1', 'exp2_3_mcar3', 'exp2_3_mcar5'],
                                                                       train_set='MCAR',
                                                                       column_name=column_name,
                                                                       metric_name=metric_name,
                                                                       db_client=db_client,
                                                                       group=group,
                                                                       base_font_size=base_font_size,
                                                                       ylim=ylim,
                                                                       with_band=with_band,
                                                                       without_dummy=without_dummy))
    print('Prepared a plot for an MCAR train set')
    _, mar_base_chart2, _ = (
        get_exp2_line_bands_for_diff_imputers_and_single_eval_scenario(dataset_name=dataset_name,
                                                                       evaluation_scenarios=['exp2_3_mar1', 'exp2_3_mar3', 'exp2_3_mar5'],
                                                                       train_set='MAR',
                                                                       column_name=column_name,
                                                                       metric_name=metric_name,
                                                                       db_client=db_client,
                                                                       group=group,
                                                                       base_font_size=base_font_size,
                                                                       ylim=ylim,
                                                                       with_band=with_band,
                                                                       without_dummy=without_dummy))
    print('Prepared a plot for an MAR train set')
    _, _, mnar_base_chart3 = (
        get_exp2_line_bands_for_diff_imputers_and_single_eval_scenario(dataset_name=dataset_name,
                                                                       evaluation_scenarios=['exp2_3_mnar1', 'exp2_3_mnar3', 'exp2_3_mnar5'],
                                                                       train_set='MNAR',
                                                                       column_name=column_name,
                                                                       metric_name=metric_name,
                                                                       db_client=db_client,
                                                                       group=group,
                                                                       base_font_size=base_font_size,
                                                                       ylim=ylim,
                                                                       with_band=with_band,
                                                                       without_dummy=without_dummy))
    print('Prepared a plot for an MNAR train set')

    # Concatenate two base charts
    main_base_chart = alt.hconcat()
    main_base_chart |= mcar_base_chart1
    main_base_chart |= mar_base_chart2
    main_base_chart |= mnar_base_chart3

    final_grid_chart = (
        main_base_chart.configure_axis(
            labelFontSize=base_font_size + 2,
            titleFontSize=base_font_size + 2,
            labelFontWeight='normal',
            titleFontWeight='normal',
        ).configure_title(
            fontSize=base_font_size + 2
        ).configure_legend(
            titleFontSize=base_font_size + 2,
            labelFontSize=base_font_size + 2,
            symbolStrokeWidth=10,
            labelLimit=400,
            titleLimit=300,
            columns=3,
            orient='top',
            direction='horizontal',
            titleAnchor='middle',
            symbolOffset=80,
        )
    )

    # Set a shared scale for the y-axis
    final_grid_chart = final_grid_chart.resolve_scale(y='shared')

    return final_grid_chart


def get_data_for_box_plots_for_diff_imputers_and_datasets(train_injection_scenario: str, test_injection_scenario: str,
                                                          metric_name: str, db_client, dataset_to_column_name: dict = None,
                                                          dataset_to_group: dict = None, without_dummy: bool = False):
    evaluation_scenario = get_evaluation_scenario(train_injection_scenario)

    imputers_metric_df_for_diff_datasets = pd.DataFrame()
    for dataset_name in DATASET_CONFIG.keys():
        group = 'overall' if dataset_to_group is None else dataset_to_group[dataset_name]
        column_name = dataset_to_column_name[dataset_name]

        metric_name = '_'.join([c.capitalize() for c in metric_name.split('_')])
        if group == 'overall':
            imputers_metric_df = get_imputers_metric_df(db_client=db_client,
                                                        dataset_name=dataset_name,
                                                        evaluation_scenario=evaluation_scenario,
                                                        column_name=column_name,
                                                        group=group)
            new_metric_name = metric_name

        else:
            imputers_metric_df = get_imputers_disparity_metric_df(db_client=db_client,
                                                                  dataset_name=dataset_name,
                                                                  evaluation_scenario=evaluation_scenario,
                                                                  column_name=column_name,
                                                                  metric_name=metric_name,
                                                                  group=group)
            imputers_metric_df['Dataset_Name'] = dataset_name
            new_metric_name = metric_name + '_Difference'

        if without_dummy:
            imputers_metric_df = imputers_metric_df[
                (imputers_metric_df['Dataset_Part'].str.contains('X_test')) &
                (imputers_metric_df['Null_Imputer_Name'] != ErrorRepairMethod.median_dummy.value)
                ]
        else:
            imputers_metric_df = imputers_metric_df[imputers_metric_df['Dataset_Part'].str.contains('X_test')]

        imputers_metric_df['Test_Injection_Scenario'] = imputers_metric_df['Dataset_Part'].apply(lambda x: x.split('_')[-1])
        imputers_metric_df = imputers_metric_df[imputers_metric_df['Test_Injection_Scenario'] == test_injection_scenario]

        imputers_metric_df_for_diff_datasets = pd.concat([imputers_metric_df_for_diff_datasets, imputers_metric_df])
        print(f'Extracted data for {dataset_name}')

    return imputers_metric_df_for_diff_datasets, new_metric_name


def create_box_plots_for_diff_imputers_and_datasets(train_injection_scenario: str, test_injection_scenario: str,
                                                    dataset_to_column_name: dict, metric_name: str,
                                                    db_client, dataset_to_group: dict = None, base_font_size: int = 22,
                                                    without_dummy: bool = False, ylim=Undefined):
    train_injection_scenario = train_injection_scenario.upper()
    test_injection_scenario = test_injection_scenario.upper()

    sns.set_style("whitegrid")
    imputers_order = ['deletion', 'median-mode', 'median-dummy', 'miss_forest',
                      'k_means_clustering', 'datawig', 'automl']
    dataset_to_sequence_num = {
        DIABETES_DATASET: 1,
        GERMAN_CREDIT_DATASET: 2,
        ACS_INCOME_DATASET: 3,
        LAW_SCHOOL_DATASET: 4,
        BANK_MARKETING_DATASET: 5,
        CARDIOVASCULAR_DISEASE_DATASET: 6,
        ACS_EMPLOYMENT_DATASET: 7,
    }
    if without_dummy:
        imputers_order = [t for t in imputers_order if t != ErrorRepairMethod.median_dummy.value]

    metric_name = '_'.join([c.capitalize() for c in metric_name.split('_')]) if 'equalized_odds' not in metric_name.lower() else metric_name
    to_plot, new_metric_name = get_data_for_box_plots_for_diff_imputers_and_datasets(train_injection_scenario=train_injection_scenario,
                                                                                     test_injection_scenario=test_injection_scenario,
                                                                                     metric_name=metric_name,
                                                                                     dataset_to_column_name=dataset_to_column_name,
                                                                                     db_client=db_client,
                                                                                     dataset_to_group=dataset_to_group,
                                                                                     without_dummy=without_dummy)

    to_plot['Dataset_Name_With_Column'] = to_plot['Dataset_Name'] + ' (' + to_plot['Column_With_Nulls'] + ')'
    to_plot['Dataset_Name_With_Column'] = to_plot['Dataset_Name_With_Column'].apply(wrap, args=[18]) # Wrap on whitespace with a max line length of 18 chars
    to_plot['Dataset_Sequence_Number'] = to_plot['Dataset_Name'].apply(lambda x: dataset_to_sequence_num[x])
    to_plot['Dataset_Name'] = to_plot['Dataset_Name'].replace({ACS_INCOME_DATASET: 'folk_inc'})

    metric_title = new_metric_name.replace('_', ' ')
    metric_title = (
        metric_title.replace('Rmse', 'RMSE')
        .replace('Kl Divergence Pred', 'KL Divergence Pred')
        .replace('Kl Divergence Total', 'KL Divergence Total')
    )

    chart = (
        alt.Chart(to_plot).mark_boxplot(
            ticks=True,
            median={'stroke': 'black', 'strokeWidth': 0.7},
        ).encode(
            x=alt.X("Null_Imputer_Name:N",
                    title=None,
                    sort=imputers_order,
                    axis=alt.Axis(labels=False)),
            y=alt.Y(f"{new_metric_name}:Q",
                    title=metric_title,
                    scale=alt.Scale(zero=True if metric_name.lower() == 'kl_divergence_pred' else False, domain=ylim)),
            color=alt.Color("Null_Imputer_Name:N", title=None, sort=imputers_order),
            column=alt.Column('Dataset_Name_With_Column',
                              title=None,
                              sort=alt.SortField(field='Dataset_Sequence_Number', order='ascending'))
        ).properties(
            width=150,
        )
    )

    final_chart = (
        chart.configure_legend(
            titleFontSize=base_font_size + 4,
            labelFontSize=base_font_size + 2,
            symbolStrokeWidth=10,
            labelLimit=400,
            titleLimit=300,
            columns=4,
            orient='top',
            direction='horizontal',
            titleAnchor='middle',
            symbolOffset=120,
        ).configure_facet(
            spacing=15 if new_metric_name.lower() == 'kl_divergence_pred' or 'difference' in new_metric_name.lower() else 5,
        ).configure_view(
            stroke=None
        ).configure_header(
            labelOrient='bottom',
            labelPadding=5,
            labelFontSize=base_font_size + 2,
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


def get_data_for_box_plots_for_diff_imputers_and_datasets_for_mixed_exp(train_injection_scenario: str, test_injection_scenario: str,
                                                                        metric_name: str, db_client, dataset_to_column_name: dict = None,
                                                                        dataset_to_group: dict = None, without_dummy: bool = False):
    evaluation_scenario = get_evaluation_scenario(train_injection_scenario)
    imputers_metric_df_for_diff_datasets = pd.DataFrame()
    for dataset_name in DATASET_CONFIG.keys():
        group = 'overall' if dataset_to_group is None else dataset_to_group[dataset_name]
        metric_name = '_'.join([c.capitalize() for c in metric_name.split('_')])
        if metric_name.lower() == 'rmse':
            column_names = dataset_to_column_name[dataset_name]['num']
        elif 'kl_divergence' in metric_name.lower():
            column_names = dataset_to_column_name[dataset_name]['cat'] + dataset_to_column_name[dataset_name]['num']
        else:
            column_names = dataset_to_column_name[dataset_name]['cat']

        for column_name in column_names:
            if group == 'overall':
                imputers_metric_df = get_imputers_metric_df(db_client=db_client,
                                                            dataset_name=dataset_name,
                                                            evaluation_scenario=evaluation_scenario,
                                                            column_name=column_name,
                                                            group=group)
                new_metric_name = metric_name

            else:
                imputers_metric_df = get_imputers_disparity_metric_df(db_client=db_client,
                                                                      dataset_name=dataset_name,
                                                                      evaluation_scenario=evaluation_scenario,
                                                                      column_name=column_name,
                                                                      metric_name=metric_name,
                                                                      group=group)
                imputers_metric_df['Dataset_Name'] = dataset_name
                new_metric_name = metric_name + '_Difference'

            if without_dummy:
                imputers_metric_df = imputers_metric_df[
                    (imputers_metric_df['Dataset_Part'].str.contains('X_test')) &
                    (imputers_metric_df['Null_Imputer_Name'] != ErrorRepairMethod.median_dummy.value)
                    ]
            else:
                imputers_metric_df = imputers_metric_df[imputers_metric_df['Dataset_Part'].str.contains('X_test')]

            imputers_metric_df['Test_Injection_Scenario'] = imputers_metric_df['Dataset_Part'].apply(lambda x: x.split('_')[-1])
            imputers_metric_df = imputers_metric_df[imputers_metric_df['Test_Injection_Scenario'] == test_injection_scenario]

            imputers_metric_df_for_diff_datasets = pd.concat([imputers_metric_df_for_diff_datasets, imputers_metric_df])

        print(f'Extracted data for {dataset_name}')

    avg_imputers_metric_df_for_diff_datasets = imputers_metric_df_for_diff_datasets.groupby(['Dataset_Name', 'Null_Imputer_Name', 'Evaluation_Scenario',
                                                                                             'Experiment_Seed', 'Dataset_Part']).mean(numeric_only=True).reset_index()
    return avg_imputers_metric_df_for_diff_datasets, new_metric_name


def create_box_plots_for_diff_imputers_and_datasets_for_mixed_exp(train_injection_scenario: str, test_injection_scenario: str,
                                                                  dataset_to_column_name: dict, metric_name: str,
                                                                  db_client, dataset_to_group: dict = None, base_font_size: int = 22,
                                                                  without_dummy: bool = False, ylim=Undefined):
    train_injection_scenario = train_injection_scenario.upper()
    test_injection_scenario = test_injection_scenario.upper()

    sns.set_style("whitegrid")
    imputers_order = ['deletion', 'median-mode', 'median-dummy', 'miss_forest',
                      'k_means_clustering', 'datawig', 'automl', 'nomi', 'tdm', 'gain', 'notmiwae']
    if without_dummy:
        imputers_order = [t for t in imputers_order if t != ErrorRepairMethod.median_dummy.value]

    metric_name = '_'.join([c.capitalize() for c in metric_name.split('_')]) if 'equalized_odds' not in metric_name.lower() else metric_name
    to_plot, new_metric_name = get_data_for_box_plots_for_diff_imputers_and_datasets_for_mixed_exp(train_injection_scenario=train_injection_scenario,
                                                                                                   test_injection_scenario=test_injection_scenario,
                                                                                                   metric_name=metric_name,
                                                                                                   dataset_to_column_name=dataset_to_column_name,
                                                                                                   db_client=db_client,
                                                                                                   dataset_to_group=dataset_to_group,
                                                                                                   without_dummy=without_dummy)
    if dataset_to_group is not None and new_metric_name.lower() == 'rmse_difference':
        to_plot = to_plot.loc[~((to_plot['Dataset_Name'] == GERMAN_CREDIT_DATASET) & (to_plot[new_metric_name] > 100))]

    dataset_to_sequence_num = {
        DIABETES_DATASET: 1,
        GERMAN_CREDIT_DATASET: 2,
        ACS_INCOME_DATASET: 3,
        LAW_SCHOOL_DATASET: 4,
        BANK_MARKETING_DATASET: 5,
        CARDIOVASCULAR_DISEASE_DATASET: 6,
        ACS_EMPLOYMENT_DATASET: 7,
    }
    to_plot['Dataset_Sequence_Number'] = to_plot['Dataset_Name'].apply(lambda x: dataset_to_sequence_num[x])
    to_plot['Dataset_Name'] = to_plot['Dataset_Name'].replace({ACS_INCOME_DATASET: 'folk_inc'})

    if 'kl_divergence' in metric_name.lower():
        to_plot['Extended_Dataset_Name'] = to_plot['Dataset_Name']
    elif 'rmse' in metric_name.lower():
        to_plot['Extended_Dataset_Name'] = to_plot['Dataset_Name'] + ' (num)'
    else:
        to_plot['Extended_Dataset_Name'] = to_plot['Dataset_Name'] + ' (cat)'

    metric_title = new_metric_name.replace('_', ' ')
    metric_title = (
        metric_title.replace('Rmse', 'RMSE')
        .replace('Kl Divergence Pred', 'KL Divergence Pred')
        .replace('Kl Divergence Total', 'KL Divergence Total')
        .replace('KL Divergence Pred Difference', 'KLD Pred Difference')
        .replace('KL Divergence Total Difference', 'KLD Total Difference')
    )

    chart = (
        alt.Chart(to_plot).mark_boxplot(
            ticks=True,
            median={'stroke': 'black', 'strokeWidth': 0.7},
        ).encode(
            x=alt.X("Null_Imputer_Name:N",
                    title=None,
                    sort=imputers_order,
                    axis=alt.Axis(labels=False)),
            y=alt.Y(f"{new_metric_name}:Q",
                    title=metric_title,
                    scale=alt.Scale(zero=True if metric_name.lower() == 'kl_divergence_pred' else False, domain=ylim)),
            color=alt.Color("Null_Imputer_Name:N", title=None, sort=imputers_order),
            column=alt.Column('Extended_Dataset_Name',
                              title=None,
                              sort=alt.SortField(field='Dataset_Sequence_Number', order='ascending'))
        ).properties(
            # height=200,
            width=190 if metric_name.lower() == 'kl_divergence_pred' else 180,
        )
    )

    final_chart = (
        chart.configure_legend(
            titleFontSize=base_font_size + 4,
            labelFontSize=base_font_size + 2,
            symbolStrokeWidth=10,
            labelLimit=400,
            titleLimit=300,
            columns=5,
            orient='top',
            direction='horizontal',
            titleAnchor='middle',
            symbolOffset=140 if dataset_to_group is not None else 110,
        ).configure_facet(
            spacing=15 if new_metric_name.lower() == 'kl_divergence_pred' or 'difference' in new_metric_name.lower() else 5,
        ).configure_view(
            stroke=None
        ).configure_header(
            labelOrient='bottom',
            labelPadding=5,
            labelFontSize=base_font_size + 2,
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
