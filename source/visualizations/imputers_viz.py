import altair as alt
import seaborn as sns
from altair.utils.schemapi import Undefined

from configs.constants import IMPUTATION_PERFORMANCE_METRICS_COLLECTION_NAME


def get_imputers_metric_df(db_client, dataset_name: str, column_name: str, group: str):
    query = {
        'dataset_name': dataset_name,
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


def create_box_plots_for_diff_imputers(dataset_name: str, column_name: str,
                                       metric_name: str, db_client,
                                       group: str = 'overall', ylim=Undefined):
    sns.set_style("whitegrid")
    base_font_size = 18
    imputers_order = ['deletion', 'median-mode', 'median-dummy', 'miss_forest',
                      'k_means_clustering', 'datawig', 'automl']

    metric_name = '_'.join([c.capitalize() for c in metric_name.split('_')])
    imputers_metric_df = get_imputers_metric_df(db_client=db_client,
                                                dataset_name=dataset_name,
                                                column_name=column_name,
                                                group=group)
    to_plot = imputers_metric_df[imputers_metric_df['Dataset_Part'].str.contains('X_test')]
    to_plot['Test_Injection_Strategy'] = to_plot['Dataset_Part'].apply(lambda x: x.split('_')[-1][:-1])
    print('to_plot.shape[0] --', to_plot.shape[0])

    metric_title = metric_name.replace('_', ' ') if metric_name.lower() != 'rmse' else 'RMSE'
    print(f'{metric_title} top 5 rows:')
    print(to_plot[metric_name].head())

    chart = (
        alt.Chart(to_plot).mark_boxplot(
            ticks=True,
            size=20,
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
        ).resolve_scale(
            x='independent'
        ).properties(
            width=180
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
        ).configure_title(
            fontSize=base_font_size + 2
        ).configure_legend(
            titleFontSize=base_font_size + 4,
            labelFontSize=base_font_size + 2,
            symbolStrokeWidth=5,
            labelLimit=400,
            titleLimit=300,
            columns=4,
            orient='top',
            direction='horizontal',
            titleAnchor='middle'
        )
    )

    return chart
