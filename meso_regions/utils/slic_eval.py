import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

path_classes = Path("/data_lids/home/taylla/PycharmProjects/meso/data"
                    "/classes_subclasses_nodular.csv")
path_images = Path("/data_lids/home/taylla/PycharmProjects/meso/data/resample"
                   "/images_orig_reg")
path_masks_dilated = Path("/data_lids/home/taylla/PycharmProjects/meso/output/"
                          "/HyperparameterTuning/Dilation")
path_metrics_train = Path("/data_lids/home/taylla/PycharmProjects/meso"
                          "/output/HyperparameterTuning/exp_dilatation_P"
                          "/dilated_patient/SLIC/split_class/train"
                          "/metrics_dilated_patient_slic_train.csv")
path_metrics_test = Path("/data_lids/home/taylla/PycharmProjects/meso/output"
                         "/HyperparameterTuning/exp_dilatation_P"
                         "/dilated_patient/SLIC/split_class/test"
                         "/metrics_dilated_patient_slic_test.csv")
path_metrics_out = Path("/data_lids/home/taylla/PycharmProjects/meso"
                            "/output/quantitative_analysis/classification/slic")
os.makedirs(path_metrics_out, exist_ok=True)
path_splits = Path("/data_lids/home/taylla/PycharmProjects/meso/data/splits")


df_slic_train = pd.read_csv(path_metrics_train, index_col=0)
df_slic_train.head()

# df_slic_test = pd.read_csv(path_metrics_test)
# #get the columns names
columns = df_slic_train.columns
#
# #get the rows names
rows = df_slic_train.index
#
print(rows)
print(columns)
# # # drop rows with all TP = 0 and TN = 0

transposed_data = df_slic_train.transpose()
# transposed_data.columns = transposed_data.iloc[
#     0]  # Set the first row as header

# Convert TP and TN to numeric type to filter properly
transposed_data['TP'] = pd.to_numeric(transposed_data['TP'])
transposed_data['TN'] = pd.to_numeric(transposed_data['TN'])

# Filter out rows where TP or TN are zero
filtered_data = transposed_data[
    (transposed_data['TP'] != 0) & (transposed_data['TN'] != 0)]
# save the filtered data
filtered_data = filtered_data.copy()

filtered_data.drop(columns=['FP_ID', 'FN_ID'], inplace=True)
print(filtered_data)
correlation_matrix = filtered_data.corr()
print(correlation_matrix)

#
#
# # df to numpy array
# df_slic_train = df_slic_train.to_numpy()
# df_slic_test = df_slic_test.to_numpy()
