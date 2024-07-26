import os

import pandas as pd
from sklearn.metrics import confusion_matrix
from pathlib import Path



path_exp = Path(
    "/data_lids/home/taylla/PycharmProjects/meso/output/HyperparameterTuning/exp")
mask = os.listdir(path_exp)
method = 'DISF'
def correct_classification_rate(y_true, y_pred):
    """
    Calculate the Correct Classification Rate.

    Parameters:
    y_true (list or array): True labels
    y_pred (list or array): Predicted labels

    Returns:
    float: Correct Classification Rate
    """
    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Calculate Correct Classification Rate
    if tp + tn + fn == 0:
        return 0  # To handle division by zero
    return (tp + tn) / (tp + tn + fn)




for m in mask:
    path_mask = path_exp / m
    path_train = path_mask / method / 'split_class' / 'train'
    df = pd.read_csv(path_train / f"metrics_{m}_{method.lower()}_train.csv",
                     index_col=0)
    df = clean_csv(df)
    dict_metrics = best_hyperparameters(df)
    df_best_metrics = pd.DataFrame(dict_metrics)
    df_best_metrics.to_csv(
        path_train / f"best_metrics_test_{m}_{method.lower()}_train.csv")