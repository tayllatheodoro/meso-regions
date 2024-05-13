# analysis per mask per method
import os
from pathlib import Path

import numpy as np
import pandas as pd

path_exp = Path(
    "/data_lids/home/taylla/PycharmProjects/meso/output/HyperparameterTuning/exp")
mask = os.listdir(path_exp)
method = 'SLIC'


def clean_csv(data, sentinel=['TN', 'TP']):
    for metric in sentinel:
        tn_row = data.loc[metric]

        # Create a list to hold the indices of columns where the value is zero
        column_indices = [i for i in range(len(tn_row)) if
                          tn_row.iloc[i] == '0']

        # Convert column indices to column labels
        column_labels = data.columns[column_indices].tolist()

        # Drop these columns from the DataFrame
        data = data.drop(columns=column_labels)

    return data


def best_hyperparameters(data, metrics=None):
    # Find the best hyperparameters
    if metrics is None:
        metrics = ['AUC', 'acc', 'balanced_acc', 'f1_score', 'Sensitivity', 'Specificity']
    best_metrics = {}
    for metric in metrics:
        best_col = data.loc[metric].idxmax()
        best_metrics[metric] = {'Value': data.at[metric, best_col],
                                'Parameters': best_col}
    return best_metrics


# # Calculate summary statistics for the entire dataset
# df_t = df.transpose()
# summary_statistics = df_t.describe(include='all').transpose()
# #savw the summary statistics to a CSV file
# summary_statistics.to_csv(path_train / f"summary_statistics_{mask}_{method.lower()}_train.csv")
#
# # Print the summary statistics
# print(summary_statistics)
for m in mask:

    path_mask = path_exp / m
    path_train = path_mask / method / 'split_class' / 'train'
    df = pd.read_csv(path_train / f"metrics_{m}_{method.lower()}_train.csv",
                     index_col=0)
    df = clean_csv(df)
    dict_metrics = best_hyperparameters(df)
    df_best_metrics = pd.DataFrame(dict_metrics)
    df_best_metrics.to_csv(
        path_train / f"best_metrics_{m}_{method.lower()}_train.csv")
