import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.mesoECE.experiment import Experiment
from src.run.config import define_config_slic, define_config_ece, \
    define_config_superspels

path_classes = Path("/data_lids/home/taylla/PycharmProjects/meso/data"
                    "/classes_subclasses_nodular.csv")
path_images = Path("/data_lids/home/taylla/PycharmProjects/meso/data/resample"
                   "/images_orig_reg")
path_output = Path("/data_lids/home/taylla/PycharmProjects/meso/output"
                   "/HyperparameterTuning/SLIC/")

path_masks_dilated = Path("/data_lids/home/taylla/PycharmProjects/meso/output/"
                          "/HyperparameterTuning/Dilation")

path_splits = Path("/data_lids/home/taylla/PycharmProjects/meso/data/splits")
path_train_metrics = Path("/data_lids/home/taylla/PycharmProjects/meso/output"
                          "/HyperparameterTuning/train_metrics")




# df_disf = pd.read_csv(path_train_metrics / "metrics_all_exp_slic.csv")
# df_fullece = pd.read_csv(path_train_metrics / "metrics_all_exp_slic.csv")

# select the row auc and select the collum with the best auc
# best_slic = df_slic.iloc[df_slic['AUC'].idxmax()]
# best_disf = df_disf.iloc[df_disf['AUC'].idxmax()]
# best_fullece = df_fullece.iloc[df_fullece['AUC'].idxmax()]
#
# print(best_slic)
# print(best_disf)
# print(best_fullece)

# # Selecting the row for AUC
# auc_row = df_slic.loc['AUC']
# best_auc_column = auc_row.idxmax()
#
# # Access the maximum AUC value
# best_auc_value = auc_row[best_auc_column]

# print(auc_row)
# print(best_auc_column)
# print(best_auc_value)

metrics_files = ["metrics_all_exp_slic.csv",
                 "metrics_all_exp_disf.csv",
                 "metrics_all_exp_fullece.csv"]

for file in metrics_files:
    df = pd.read_csv(path_train_metrics / file, index_col=0)
    auc_row = df.loc['AUC']
    best_auc_column = auc_row.idxmax()
    best_auc_value = auc_row[best_auc_column]

    acc_row = df.loc['acc']
    best_acc_column = acc_row.idxmax()
    best_acc_value = acc_row[best_acc_column]

    f1_row = df.loc['f1_score']
    best_f1_column = f1_row.idxmax()
    best_f1_value = f1_row[best_f1_column]
    method = file.split('_')[3].split('.')[0]
    best_metrics = {
        'AUC': {'Value': best_auc_value, 'Parameters': best_auc_column},
        'ACC': {'Value': best_acc_value, 'Parameters': best_acc_column},
        'F1': {'Value': best_f1_value, 'Parameters': best_f1_column}}

    df_best_metrics = pd.DataFrame(best_metrics)
    print(df_best_metrics)
    df_best_metrics.to_csv(path_train_metrics / f"best_metrics_{method}.csv")




