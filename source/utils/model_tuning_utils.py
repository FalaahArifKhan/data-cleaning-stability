import os
import getpass
import shutil
import pathlib
import lightgbm
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pprint import pprint, pformat
from copy import deepcopy
from datetime import datetime

from pytorch_lightning import seed_everything
from pytorch_tabular.config import DataConfig, TrainerConfig, OptimizerConfig, ModelConfig
from pytorch_tabular.tabular_model_tuner import TabularModelTuner
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer, accuracy_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

from virny.custom_classes.base_dataset import BaseFlowDataset
from source.custom_classes.grid_search_cv_with_early_stopping import GridSearchCVWithEarlyStopping


def tune_sklearn_models(models_params_for_tuning: dict, base_flow_dataset: BaseFlowDataset,
                        dataset_name: str, n_folds: int = 3):
    """
    Tune each model on a validation set with GridSearchCV.

    Return each model with its best hyperparameters that have the highest F1 score and Accuracy.
     results_df is a dataframe with metrics and tuned parameters;
     models_config is a dict with model tuned params for the metrics computation stage
    """
    models_config = dict()
    tuned_params_df = pd.DataFrame(columns=('Dataset_Name', 'Model_Name', 'F1_Score', 'Accuracy_Score', 'Runtime_In_Mins', 'Model_Best_Params'))
    # Find the most optimal hyperparameters based on accuracy and F1-score for each model in models_config
    for model_idx, (model_name, model_params) in enumerate(models_params_for_tuning.items()):
        try:
            tuning_start_time = datetime.now()
            print(f"{tuning_start_time.strftime('%Y/%m/%d, %H:%M:%S')}: Tuning {model_name}...", flush=True)
            cur_model, cur_f1_score, cur_accuracy, cur_params = validate_model(deepcopy(model_params['model']),
                                                                               base_flow_dataset.X_train_val,
                                                                               base_flow_dataset.y_train_val,
                                                                               model_params['params'],
                                                                               n_folds)
            tuning_end_time = datetime.now()
            print(f'{tuning_end_time.strftime("%Y/%m/%d, %H:%M:%S")}: Tuning for {model_name} is finished '
                  f'[F1 score = {cur_f1_score}, Accuracy = {cur_accuracy}]\n', flush=True)
            print(f'Best hyper-parameters for {model_name}:\n{pformat(cur_params)}', flush=True)

        except Exception as err:
            print(f"ERROR with {model_name}: ", err)
            continue

        # Save test results of each model in dataframe
        tuning_duration = (tuning_end_time - tuning_start_time).total_seconds() / 60.0
        tuned_params_df.loc[model_idx] = [dataset_name, model_name, cur_f1_score, cur_accuracy, tuning_duration, cur_params]
        models_config[model_name] = model_params['model'].set_params(**cur_params)

    return tuned_params_df, models_config


def tune_pytorch_tabular_models(models_params_for_tuning: dict, base_flow_dataset: BaseFlowDataset, dataset_name: str,
                                null_imputer_name: str, evaluation_scenario: str, experiment_seed: int):
    """
    Tune each defined model from pytorch tabular on a validation set.
    """
    models_config = dict()
    tuned_params_df = pd.DataFrame(columns=('Dataset_Name', 'Model_Name', 'F1_Score', 'Accuracy_Score', 'Runtime_In_Mins', 'Model_Best_Params'))
    # Find the most optimal hyperparameters based on accuracy and F1-score for each model in models_config
    for model_idx, (model_name, model_params) in enumerate(models_params_for_tuning.items()):
        tuning_start_time = datetime.now()
        print(f"{tuning_start_time.strftime('%Y/%m/%d, %H:%M:%S')}: Tuning {model_name}...", flush=True)

        saved_models_prefix = os.path.join(model_name, null_imputer_name, dataset_name, evaluation_scenario)
        cur_model, cur_f1_score, cur_accuracy, cur_params = validate_pytorch_tabular_model(model_config=model_params['model'],
                                                                                           optimizer_config=model_params['optimizer_config'],
                                                                                           trainer_config=model_params['trainer_config'],
                                                                                           search_space=model_params['params'],
                                                                                           base_flow_dataset=base_flow_dataset,
                                                                                           saved_models_prefix=saved_models_prefix,
                                                                                           experiment_seed=experiment_seed)
        tuning_end_time = datetime.now()
        print(f'{tuning_end_time.strftime("%Y/%m/%d, %H:%M:%S")}: Tuning for {model_name} is finished '
              f'[F1 score = {cur_f1_score}, Accuracy = {cur_accuracy}]\n', flush=True)
        print(f'Best hyper-parameters for {model_name}:\n{pformat(cur_params)}', flush=True)

        # Save test results of each model in dataframe
        tuning_duration = (tuning_end_time - tuning_start_time).total_seconds() / 60.0
        tuned_params_df.loc[model_idx] = [dataset_name, model_name, cur_f1_score, cur_accuracy, tuning_duration, cur_params]
        models_config[model_name] = cur_model

    return tuned_params_df, models_config


def validate_model(model, x, y, params, n_folds):
    """
    Use GridSearchCV for a special model to find the best hyperparameters based on validation set
    """
    if isinstance(model, lightgbm.LGBMClassifier):
        grid_search_cv_class = GridSearchCVWithEarlyStopping
    else:
        grid_search_cv_class = GridSearchCV

    grid_search = grid_search_cv_class(estimator=model,
                                       param_grid=params,
                                       scoring={
                                           "F1_Score": make_scorer(f1_score, average='macro'),
                                           "Accuracy_Score": make_scorer(accuracy_score),
                                       },
                                       refit="F1_Score",
                                       n_jobs=-1,
                                       cv=n_folds,
                                       verbose=0)
    grid_search.fit(x, y.values.ravel())
    best_index = grid_search.best_index_

    return grid_search.best_estimator_, \
           grid_search.cv_results_["mean_test_F1_Score"][best_index], \
           grid_search.cv_results_["mean_test_Accuracy_Score"][best_index], \
           grid_search.best_params_


def validate_pytorch_tabular_model(model_config: ModelConfig, optimizer_config: OptimizerConfig,
                                   trainer_config: TrainerConfig, search_space: dict,
                                   base_flow_dataset, saved_models_prefix, experiment_seed: int):
    """
    Use GridSearchCV for a special model from pytorch tabular
     to find the best hyperparameters based on the validation set
    """
    # Prepare train and val sets
    train_val = pd.concat([base_flow_dataset.X_train_val, base_flow_dataset.y_train_val], axis=1)
    train, val = train_test_split(train_val, random_state=experiment_seed, test_size=base_flow_dataset.y_test.shape[0])

    # Create a scorer function
    macro_f1_score = lambda y_true, y_pred: f1_score(y_true.values.ravel(), y_pred['prediction'].values.ravel(), average='macro')
    macro_f1_score.__name__ = 'f1_score'

    # Initialize a tabular model tuner
    data_config = DataConfig(
        target=[base_flow_dataset.y_train_val.name],
        continuous_cols=[col for col in base_flow_dataset.X_train_val.columns if col.startswith('num_')],
        categorical_cols=[col for col in base_flow_dataset.X_train_val.columns if col.startswith('cat_')],
    )

    saved_models_path = (pathlib.Path(__file__).parent.parent.parent
                           .joinpath('results')
                           .joinpath('intermediate_state')
                           .joinpath('saved_models')
                           .joinpath(saved_models_prefix)
                           .joinpath(str(experiment_seed)))
    if getpass.getuser() in ('dh3553', 'np2969'):
        # Use bigger storage on the HPC cluster
        saved_models_path = str(saved_models_path).replace('home', 'scratch')
    tuner = TabularModelTuner(
        data_config=data_config,
        model_config=model_config,
        optimizer_config=optimizer_config,
        trainer_config=trainer_config,
        suppress_lightning_logger=True,
    )
    tuner.trainer_config.checkpoints_path = saved_models_path

    seed_everything(seed=experiment_seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = tuner.tune(
            train=train,
            validation=val,
            search_space=search_space,
            strategy="random_search",
            # n_trials=25 if base_flow_dataset.X_train_val.shape[0] > 20_000 or 'folk' in saved_models_prefix else 100,
            n_trials=100,
            metric=macro_f1_score,
            mode="max",
            progress_bar=True,
            random_state=experiment_seed,
            verbose=False # Make True if you want to log metrics and params each iteration
        )

    # Remove all files created by TabularModelTuner to save storage space
    shutil.rmtree(saved_models_path, ignore_errors=True)
    shutil.rmtree(pathlib.Path(__file__).parent.parent.parent.joinpath('lightning_logs'), ignore_errors=True)

    return (result.best_model,
            result.best_score,
            None, # For Accuracy. Since Pytorch Tabular cannot compute two metrics during tuning,
                  # but we need accuracy for consistency with the sklearn API.
            result.best_params)


def test_evaluation(cur_best_model, model_name, cur_best_params,
                    cur_x_train, cur_y_train, cur_x_test, cur_y_test,
                    dataset_title, show_plots, debug_mode):
    """
    Evaluate model on test set.

    :return: F1 score, accuracy and predicted values, which we use to visualisations for model comparison later.
    """
    cur_best_model.fit(cur_x_train, cur_y_train.values.ravel()) # refit model on the whole train set
    cur_model_pred = cur_best_model.predict(cur_x_test)
    test_f1_score = f1_score(cur_y_test, cur_model_pred, average='macro')
    test_accuracy = accuracy_score(cur_y_test, cur_model_pred)

    if debug_mode:
        print("#" * 20, f' {dataset_title} ', "#" * 20)
        print('Test model: ', model_name)
        print('Test model parameters:')
        pprint(cur_best_params)

        # print the scores
        print()
        print(classification_report(cur_y_test, cur_model_pred, digits=3))

    if show_plots:
        # plot the confusion matrix
        sns.set_style("white")
        cm = confusion_matrix(cur_y_test, cur_model_pred, labels=cur_best_model.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Employed", "Not Employed"])
        disp.plot()
        plt.show()

    return test_f1_score, test_accuracy, cur_model_pred


def test_ML_models(best_results_df, models_config, n_folds, X_train, y_train, X_test, y_test,
                   dataset_title, show_plots, debug_mode):
    """
    Find the best model from defined list.
    Tune each model on a validation set with GridSearchCV and
    return best_model with its hyperparameters, which has the highest F1 score
    """
    results_df = pd.DataFrame(columns=('Dataset_Name', 'Model_Name', 'F1_Score',
                                       'Accuracy_Score',
                                       'Model_Best_Params'))
    best_f1_score = -np.Inf
    best_accuracy = -np.Inf
    best_model_pred = []
    best_model_name = 'No model'
    best_params = None
    idx = 0
    # find the best model among defined in models_config
    for model_config in models_config:
        try:
            print(f"{datetime.now().strftime('%Y/%m/%d, %H:%M:%S')}: Tuning {model_config['model_name']}...")
            cur_model, cur_f1_score, cur_accuracy, cur_params = validate_model(deepcopy(model_config['model']),
                                                                               X_train, y_train, model_config['params'],
                                                                               n_folds)
            print(f'{datetime.now().strftime("%Y/%m/%d, %H:%M:%S")}: Tuning for {model_config["model_name"]} is finished')

            test_f1_score, test_accuracy, cur_model_pred = test_evaluation(cur_model, model_config['model_name'], cur_params,
                                                                           X_train, y_train, X_test, y_test, dataset_title, show_plots, debug_mode)
        except Exception as err:
            print(f"ERROR with {model_config['model_name']}: ", err)
            continue

        # save test results of each model in dataframe
        results_df.loc[idx] = [dataset_title,
                               model_config['model_name'],
                               test_f1_score,
                               test_accuracy,
                               cur_params]
        idx += 1

        if test_f1_score > best_f1_score:
            best_f1_score = test_f1_score
            best_accuracy = test_accuracy
            best_model_name = model_config['model_name']
            best_params = cur_params
            best_model_pred = cur_model_pred

    # append results of best model in best_results_df
    best_results_df.loc[best_results_df.shape[0]] = [dataset_title,
                                                     best_model_name,
                                                     best_f1_score,
                                                     best_accuracy,
                                                     best_params,
                                                     best_model_pred]

    return results_df, best_results_df
