import altair as alt
import seaborn as sns
from altair.utils.schemapi import Undefined

from configs.constants import IMPUTATION_PERFORMANCE_METRICS_COLLECTION_NAME, ErrorRepairMethod


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

    # Check uniqueness
    duplicates_mask = metric_df.duplicated(subset=['Imputation_Guid'], keep=False)
    assert len(metric_df[duplicates_mask]) == 0, 'Metric df contains duplicates'

    return metric_df


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
    imputers_metric_df = get_imputers_metric_df(db_client=db_client,
                                                dataset_name=dataset_name,
                                                evaluation_scenario=evaluation_scenario,
                                                column_name=column_name,
                                                group=group)
    if without_dummy:
        to_plot = imputers_metric_df[
                    (imputers_metric_df['Dataset_Part'].str.contains('X_test')) &
                    (imputers_metric_df['Null_Imputer_Name'] != ErrorRepairMethod.median_dummy.value)
                  ]
    else:
        to_plot = imputers_metric_df[imputers_metric_df['Dataset_Part'].str.contains('X_test')]

    to_plot['Test_Injection_Strategy'] = to_plot['Dataset_Part'].apply(lambda x: x.split('_')[-1][:-1])

    if metric_name.lower() == 'rmse':
        metric_title = 'RMSE'
    elif metric_name.lower() == 'kl_divergence_pred':
        metric_title = 'KL divergence pred'
    elif metric_name.lower() == 'kl_divergence_total':
        metric_title = 'KL divergence total'
    else:
        metric_title = metric_name.replace('_', ' ') if metric_name.lower() != 'rmse' else 'RMSE'

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
    main_base_chart = alt.vconcat()
    row = alt.hconcat()
    row |= base_chart1
    row |= base_chart2
    row |= base_chart3
    main_base_chart &= row

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
            symbolOffset=110,
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
