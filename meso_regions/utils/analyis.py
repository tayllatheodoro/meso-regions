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
path_metrics_results = Path("/data_lids/home/taylla/PycharmProjects/meso"
                            "/output/HyperparameterTuning/exp_dilatation_P"
                            "/dilated_patient/split1")
path_metrics_out = Path("/data_lids/home/taylla/PycharmProjects/meso"
                            "/output/HyperparameterTuning/exp_dilatation_P"
                            "/dilated_patient/split1/test")
os.makedirs(path_metrics_out, exist_ok=True)
path_splits = Path("/data_lids/home/taylla/PycharmProjects/meso/data/splits")

list_results = os.listdir(path_metrics_results)

for result in list_results:
    if 'slic_train' in result:
        df_slic_train = pd.read_csv(path_metrics_results / result)
        # drop rows with all TP = 0 and TN = 0
        # Transpose the data
        transposed_data = df_slic_train.transpose()
        transposed_data.columns = transposed_data.iloc[
            0]  # Set the first row as header
        transposed_data = transposed_data.drop(transposed_data.index[0])

        # Convert TP and TN to numeric type to filter properly
        transposed_data['TP'] = pd.to_numeric(transposed_data['TP'])
        transposed_data['TN'] = pd.to_numeric(transposed_data['TN'])

        # Filter out rows where TP or TN are zero
        filtered_data = transposed_data[
            (transposed_data['TP'] != 0) & (transposed_data['TN'] != 0)]
        # save the filtered data
        filtered_data = filtered_data.copy()

        filtered_data.drop(columns=['FP_ID', 'FN_ID'], inplace=True)

        # filtered_data.to_csv(path_metrics_out/'filtered_data.csv', index=True)
        # transposed_data.to_csv(path_metrics_out / 'transposed_data.csv', index=True)

        # Calculate the correlation matrix
        correlation_matrix = filtered_data.corr()

        # Extract correlations related to FN
        fn_correlations = correlation_matrix['FN']

        # Plot the correlations with FN
        plt.figure(figsize=(10, 6))
        fn_correlations.drop('FN').plot(kind='bar', color='b')
        plt.title('Correlation of Metrics with False Negatives (FN)')
        plt.xlabel('Metrics')
        plt.ylabel('Correlation Coefficient')
        plt.grid(True)
        plt.show()

        # Plot the correlation matrix using a heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Correlation Matrix')
        plt.show()

        # Print the correlations
        print(fn_correlations)

        # Identify Best Configurations based on F1 Score
        best_f1_configurations = filtered_data.sort_values(by='f1_score',
                                                  ascending=False).head(10)
        # save the best configurations
        best_f1_configurations.to_csv(path_metrics_out / 'best_f1_configurations.csv', index=True)

        # Identify Best Configurations based on AUC
        best_auc_configurations = filtered_data.sort_values(by='AUC',
                                                   ascending=False).head(10)
        best_auc_configurations.to_csv(
            path_metrics_out / 'best_auc_configurations.csv', index=True)
        # print(best_f1_configurations)
        print(best_auc_configurations)

        # Scatter Plot for Sensitivity vs. Specificity
        plt.figure(figsize=(10, 8))
        sns.scatterplot(data=filtered_data, x='Sensitivity', y='Specificity', hue='AUC',
                        palette='viridis', size='AUC', sizes=(50, 200))
        plt.title('Sensitivity vs Specificity, colored by AUC')
        plt.xlabel('Sensitivity')
        plt.ylabel('Specificity')
        plt.legend(title='AUC Rating', loc='upper left')
        plt.grid(True)
        plt.show()

        # drop columns with all TP = 0 and TN = 0
    elif 'slic_test' in result:
        df_slic_test = pd.read_csv(path_metrics_results / result)
    elif 'disf_train' in result:
        df_disf_train = pd.read_csv(path_metrics_results/ result)
    elif 'disf_test' in result:
        df_disf_test = pd.read_csv(path_metrics_results / result)




