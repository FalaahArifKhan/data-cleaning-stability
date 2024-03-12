import numpy as np
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt


def set_default_plot_properties():
    plt.style.use('mpl20')
    matplotlib.rcParams['figure.dpi'] = 100
    matplotlib.rcParams['figure.figsize'] = 15, 5


def sns_set_size(height, width):
    sns.set(rc={'figure.figsize':(width, height)})


def get_proportions(protected_groups, X_data):
    for col_name in protected_groups.keys():
        proportion = protected_groups[col_name].shape[0] / X_data.shape[0]
        print(f'{col_name}: {round(proportion, 3)}')


def get_base_rates(protected_groups, y_data):
    for col_name in protected_groups.keys():
        filtered_df = y_data.iloc[protected_groups[col_name].index].copy(deep=True)
        base_rate = filtered_df[filtered_df == 1].shape[0] / filtered_df.shape[0]
        print(f'{col_name}: {round(base_rate, 3)}')

    base_rate = y_data[y_data == 1].shape[0] / y_data.shape[0]
    print(f'overall: {round(base_rate, 3)}')


def get_correlation_with_target(df, target_name, feature_names, method='spearman', heatmap_size=(16, 15)):
    # Look at the feature correlation with target
    filtered_df = df[feature_names + [target_name]]

    sns_set_size(height=heatmap_size[0], width=heatmap_size[1])
    ax = plt.axes()
    sns.heatmap(
        filtered_df.corr(method=method)[[target_name]] \
            .sort_values(by=target_name, ascending=False),
        ax=ax,
        annot=True
    )
    ax.set_title(f'{method.capitalize()} correlation')
    plt.yticks(rotation=0)
    plt.show()

    set_default_plot_properties()


def get_correlation_matrix(df, feature_names, method='spearman', heatmap_size=(16, 15)):
    # Look at correlations among features
    filtered_df = df[feature_names]

    sns_set_size(height=heatmap_size[0], width=heatmap_size[1])
    ax = plt.axes()
    sns.heatmap(filtered_df.corr(method=method), ax=ax, annot=True)
    ax.set_title(f'{method.capitalize()} Correlation')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.show()

    set_default_plot_properties()


def get_features_by_correlation_threshold(df, threshold, method='spearman'):
    # Create correlation matrix
    corr_matrix = df.corr(method=method).abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Find features with correlation greater than 0.95
    filtered_features = [column for column in upper.columns if any(upper[column] >= threshold)]

    return filtered_features


def get_features_by_target_correlation_threshold(df, target, threshold, method='spearman'):
    # Create correlation matrix
    corr_matrix = df.corr(method=method)[[target]].abs()
    feature_names = corr_matrix[corr_matrix[target] >= threshold].index

    return [col for col in feature_names if col != target]
