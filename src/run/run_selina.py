from pathlib import Path

import pandas as pd
from sklearn import metrics

path_classes_selina = Path("/data_lids/home/taylla/PycharmProjects/meso/data"
                    "/classes_subclasses_nodular_ST.csv")
path_classes = Path("/data_lids/home/taylla/PycharmProjects/meso/data"
                    "/classes_subclasses_nodular.csv")
#
df_selina = pd.read_csv(path_classes_selina)
df_classes = pd.read_csv(path_classes)
# convert the dfs to dict
dict_selina = dict(df_selina)
dict_classes = dict(df_classes)
# print(df_classes)
#  select the CLASS and ID columns for each df
df_selina = df_selina[['ID', 'CLASS']]
df_classes = df_classes[['ID', 'CLASS']]
print(df_selina)
metrics_exp = {}
ids_fp = []
ids_fn = []

y = df_classes['CLASS'].tolist()
y_pred = df_selina['CLASS'].tolist()

# # add IDS of false positive and false negative to lists
# for result_ece in results_ece:
#     if result_ece[1] == 1 and result_ece[2] == 0:
#         ids_fn.append(result_ece[0])
#     if result_ece[1] == 0 and result_ece[2] == 1:
#         ids_fp.append(result_ece[0])

metrics_exp['acc'] = metrics.accuracy_score(y, y_pred)
metrics_exp['balanced_acc'] = metrics.balanced_accuracy_score(y, y_pred)
metrics_exp[
    'average_precision_score'] = metrics.average_precision_score(y,
                                                                 y_pred)
metrics_exp['f1_score'] = metrics.f1_score(y, y_pred)
metrics_exp['AUC'] = metrics.roc_auc_score(y, y_pred)
metrics_exp['Sensitivity'] = metrics.recall_score(y, y_pred)
metrics_exp['Specificity'] = metrics.precision_score(y, y_pred)

tn, fp, fn, tp = metrics.confusion_matrix(y, y_pred).ravel()
metrics_exp['PPV'] = tp / (tp + fp)
metrics_exp['NPV'] = tn / (tn + fn)
metrics_exp['FP'] = fp
metrics_exp['FN'] = fn
metrics_exp['TN'] = tn
metrics_exp['TP'] = tp
metrics_exp['FP_ID'] = ids_fp
metrics_exp['FN_ID'] = ids_fn


df_metrics = pd.DataFrame.from_dict(metrics_exp,
                                    orient='index',
                                    columns=['Results'])
df_metrics.to_csv(str('metrics.csv'))
print(df_metrics)
