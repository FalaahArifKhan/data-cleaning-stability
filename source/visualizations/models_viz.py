import altair as alt
import seaborn as sns
from altair.utils.schemapi import Undefined

from configs.constants import EXP_COLLECTION_NAME
from configs.scenarios_config import EVALUATION_SCENARIOS_CONFIG


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

    # Check uniqueness
    duplicates_mask = metric_df.duplicated(subset=['Exp_Pipeline_Guid', 'Model_Name', 'Subgroup', 'Metric', 'Test_Set_Index'], keep=False)
    assert len(metric_df[duplicates_mask]) == 0, 'Metric df contains duplicates'

    columns_subset = ['Dataset_Name', 'Null_Imputer_Name', 'Evaluation_Scenario', 'Virny_Random_State',
                      'Model_Name', 'Subgroup', 'Metric', 'Metric_Value', 'Test_Set_Index']
    metric_df = metric_df[columns_subset]

    return metric_df


def create_scatter_plots_for_diff_models_and_single_eval_scenario(dataset_name: str, evaluation_scenario: str,
                                                                  metric_name: str, db_client, title: str,
                                                                  group: str = 'overall', base_font_size: int = 18,
                                                                  ylim=Undefined):
    sns.set_style("whitegrid")
    imputers_order = ['deletion', 'median-mode', 'median-dummy', 'miss_forest',
                      'k_means_clustering', 'datawig', 'automl']

    metric_name = '_'.join([c.capitalize() for c in metric_name.split('_')])
    models_metric_df = get_models_metric_df(db_client=db_client,
                                            dataset_name=dataset_name,
                                            evaluation_scenario=evaluation_scenario,
                                            metric_name=metric_name,
                                            group=group)
    # Group metric values by seed
    models_metric_df = models_metric_df.groupby(['Dataset_Name', 'Null_Imputer_Name', 'Evaluation_Scenario',
                                                 'Model_Name', 'Subgroup', 'Metric', 'Test_Set_Index']).mean().reset_index()

    models_metric_df['Test_Injection_Strategy'] = models_metric_df.apply(
        lambda row: EVALUATION_SCENARIOS_CONFIG[row['Evaluation_Scenario']]['test_injection_scenarios'][row['Test_Set_Index']][:-1],
        axis=1
    )
    metric_title = metric_name.replace('_', ' ')
    chart = (
        alt.Chart(models_metric_df).mark_circle(
            size=60
        ).encode(
            x=alt.X("Null_Imputer_Name:N",
                    title=None,
                    sort=imputers_order,
                    axis=alt.Axis(labels=False)),
            y=alt.Y(f"Metric_Value:Q",
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


def create_scatter_plots_for_diff_models(dataset_name: str, metric_name: str, db_client,
                                         group: str = 'overall', ylim=Undefined):
    base_font_size = 20
    base_chart1 = create_scatter_plots_for_diff_models_and_single_eval_scenario(dataset_name=dataset_name,
                                                                                evaluation_scenario='exp1_mcar3',
                                                                                title='MCAR train set',
                                                                                metric_name=metric_name,
                                                                                db_client=db_client,
                                                                                group=group,
                                                                                base_font_size=base_font_size,
                                                                                ylim=ylim)
    base_chart2 = create_scatter_plots_for_diff_models_and_single_eval_scenario(dataset_name=dataset_name,
                                                                                evaluation_scenario='exp1_mar3',
                                                                                title='MAR train set',
                                                                                metric_name=metric_name,
                                                                                db_client=db_client,
                                                                                group=group,
                                                                                base_font_size=base_font_size,
                                                                                ylim=ylim)
    base_chart3 = create_scatter_plots_for_diff_models_and_single_eval_scenario(dataset_name=dataset_name,
                                                                                evaluation_scenario='exp1_mnar3',
                                                                                title='MNAR train set',
                                                                                metric_name=metric_name,
                                                                                db_client=db_client,
                                                                                group=group,
                                                                                base_font_size=base_font_size,
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


def create_box_plots_for_diff_models_and_single_eval_scenario(dataset_name: str, evaluation_scenario: str,
                                                              null_imputer_name: str, metric_name: str, db_client,
                                                              title: str, group: str = 'overall',
                                                              base_font_size: int = 18, ylim=Undefined):
    sns.set_style("whitegrid")
    models_order = ['dt_clf', 'lr_clf', 'lgbm_clf', 'rf_clf', 'mlp_clf']

    metric_name = '_'.join([c.capitalize() for c in metric_name.split('_')])
    models_metric_df = get_models_metric_df(db_client=db_client,
                                            dataset_name=dataset_name,
                                            evaluation_scenario=evaluation_scenario,
                                            metric_name=metric_name,
                                            group=group)
    models_metric_df = models_metric_df[models_metric_df['Null_Imputer_Name'] == null_imputer_name]
    models_metric_df['Test_Injection_Strategy'] = models_metric_df.apply(
        lambda row: EVALUATION_SCENARIOS_CONFIG[row['Evaluation_Scenario']]['test_injection_scenarios'][row['Test_Set_Index']][:-1],
        axis=1
    )

    metric_title = metric_name.replace('_', ' ')
    chart = (
        alt.Chart(models_metric_df).mark_boxplot(
            ticks=True,
            median={'stroke': 'black', 'strokeWidth': 0.7},
        ).encode(
            x=alt.X("Model_Name:N",
                    title=None,
                    sort=models_order,
                    axis=alt.Axis(labels=False)),
            y=alt.Y(f"Metric_Value:Q",
                    title=metric_title,
                    scale=alt.Scale(zero=False, domain=ylim)),
            color=alt.Color("Model_Name:N", title=None, sort=models_order),
            column=alt.Column('Test_Injection_Strategy:N',
                              title=None,
                              sort=['MCAR', 'MAR', 'MNAR'])
        ).properties(
            width=120,
            title=alt.TitleParams(text=title, fontSize=base_font_size + 6, anchor='middle', align='center', dx=40),
        )
    )

    return chart


def create_box_plots_for_diff_models(dataset_name: str, null_imputer_name: str, metric_name: str, db_client,
                                     group: str = 'overall', ylim=Undefined):
    base_font_size = 20
    base_chart1 = create_box_plots_for_diff_models_and_single_eval_scenario(dataset_name=dataset_name,
                                                                            evaluation_scenario='exp1_mcar3',
                                                                            title='MCAR train set',
                                                                            null_imputer_name=null_imputer_name,
                                                                            metric_name=metric_name,
                                                                            db_client=db_client,
                                                                            group=group,
                                                                            base_font_size=base_font_size,
                                                                            ylim=ylim)
    base_chart2 = create_box_plots_for_diff_models_and_single_eval_scenario(dataset_name=dataset_name,
                                                                            evaluation_scenario='exp1_mar3',
                                                                            title='MAR train set',
                                                                            null_imputer_name=null_imputer_name,
                                                                            metric_name=metric_name,
                                                                            db_client=db_client,
                                                                            group=group,
                                                                            base_font_size=base_font_size,
                                                                            ylim=ylim)
    base_chart3 = create_box_plots_for_diff_models_and_single_eval_scenario(dataset_name=dataset_name,
                                                                            evaluation_scenario='exp1_mnar3',
                                                                            title='MNAR train set',
                                                                            null_imputer_name=null_imputer_name,
                                                                            metric_name=metric_name,
                                                                            db_client=db_client,
                                                                            group=group,
                                                                            base_font_size=base_font_size,
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
            columns=5,
            orient='top',
            direction='horizontal',
            titleAnchor='middle',
            symbolOffset=150,
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


def create_box_plots_for_diff_imputers_and_single_eval_scenario_v2(dataset_name: str, evaluation_scenario: str,
                                                                   model_name: str, metric_name: str, db_client,
                                                                   title: str, group: str = 'overall',
                                                                   base_font_size: int = 18, ylim=Undefined):
    sns.set_style("whitegrid")
    imputers_order = ['deletion', 'median-mode', 'median-dummy', 'miss_forest',
                      'k_means_clustering', 'datawig', 'automl']

    metric_name = '_'.join([c.capitalize() for c in metric_name.split('_')])
    models_metric_df = get_models_metric_df(db_client=db_client,
                                            dataset_name=dataset_name,
                                            evaluation_scenario=evaluation_scenario,
                                            metric_name=metric_name,
                                            group=group)
    models_metric_df = models_metric_df[models_metric_df['Model_Name'] == model_name]
    models_metric_df['Test_Injection_Strategy'] = models_metric_df.apply(
        lambda row: EVALUATION_SCENARIOS_CONFIG[row['Evaluation_Scenario']]['test_injection_scenarios'][row['Test_Set_Index']][:-1],
        axis=1
    )

    metric_title = metric_name.replace('_', ' ')
    chart = (
        alt.Chart(models_metric_df).mark_boxplot(
            ticks=True,
            median={'stroke': 'black', 'strokeWidth': 0.7},
        ).encode(
            x=alt.X("Null_Imputer_Name:N",
                    title=None,
                    sort=imputers_order,
                    axis=alt.Axis(labels=False)),
            y=alt.Y(f"Metric_Value:Q",
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


def create_box_plots_for_diff_imputers_v2(dataset_name: str, model_name: str, metric_name: str, db_client,
                                          group: str = 'overall', ylim=Undefined):
    base_font_size = 20
    base_chart1 = create_box_plots_for_diff_imputers_and_single_eval_scenario_v2(dataset_name=dataset_name,
                                                                                 evaluation_scenario='exp1_mcar3',
                                                                                 title='MCAR train set',
                                                                                 model_name=model_name,
                                                                                 metric_name=metric_name,
                                                                                 db_client=db_client,
                                                                                 group=group,
                                                                                 base_font_size=base_font_size,
                                                                                 ylim=ylim)
    base_chart2 = create_box_plots_for_diff_imputers_and_single_eval_scenario_v2(dataset_name=dataset_name,
                                                                                 evaluation_scenario='exp1_mar3',
                                                                                 title='MAR train set',
                                                                                 model_name=model_name,
                                                                                 metric_name=metric_name,
                                                                                 db_client=db_client,
                                                                                 group=group,
                                                                                 base_font_size=base_font_size,
                                                                                 ylim=ylim)
    base_chart3 = create_box_plots_for_diff_imputers_and_single_eval_scenario_v2(dataset_name=dataset_name,
                                                                                 evaluation_scenario='exp1_mnar3',
                                                                                 title='MNAR train set',
                                                                                 model_name=model_name,
                                                                                 metric_name=metric_name,
                                                                                 db_client=db_client,
                                                                                 group=group,
                                                                                 base_font_size=base_font_size,
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
