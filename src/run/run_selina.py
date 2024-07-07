import os
from pathlib import Path

import pandas as pd
from sklearn import metrics

from src.mesoECE.experiment import correct_classification_rate

path_classes_selina = Path('/data_lids/home/taylla/PycharmProjects/meso/data'
                           '/classes_subclasses_nodular_ST.csv')
path_classes = Path("/data_lids/home/taylla/PycharmProjects/meso/data"
                    "/classes_subclasses_nodular.csv")
path_output = Path("/data_lids/home/taylla/PycharmProjects/meso/output/Selina")
os.makedirs(path_output, exist_ok=True)
path_splits = Path("/data_lids/home/taylla/PycharmProjects/meso/data/splits")

# for s in os.listdir(path_splits):

if True:
    df_selina = pd.read_csv(path_classes_selina)
    df_classes = pd.read_csv(path_classes)
    # convert the dfs to dict
    dict_selina = dict(df_selina)
    dict_classes = dict(df_classes)
    # s_n = s.split('_')[-1].split('.')[0]
    # mode = s.split('_')[0]

    ids = list(sorted(
        (pd.read_csv(path_classes)["ID"]).to_list()))
    threads = min(os.cpu_count(), len(ids))
    df_selina = df_selina[['ID', 'CLASS']]
    df_classes = df_classes[['ID', 'CLASS']]
    print(df_selina)
    metrics_exp = {}
    ids_fp = []
    ids_fn = []
    df_classes.set_index('ID', inplace=True)
    df_selina.set_index('ID', inplace=True)
    y = []
    y_pred = []
    FP = []
    FN = []
    for id in ids:
        y.append(df_classes.loc[id, 'CLASS'])
        y_pred.append(df_selina.loc[id, 'CLASS'])
        if y_pred[-1] != y[-1]:
            if y_pred[-1] == 1:
                FP.append(id)

            else:
                FN.append(id)
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
    if (tp + fp) > 0:
        metrics_exp['PPV'] = tp / (tp + fp)
    else:
        metrics_exp['PPV'] = 0
    if (tn + fn) > 0:
        metrics_exp['NPV'] = tn / (tn + fn)
    else:
        metrics_exp['NPV'] = 0
    metrics_exp['FP'] = fp
    metrics_exp['FN'] = fn
    metrics_exp['TN'] = tn
    metrics_exp['TP'] = tp
    metrics_exp['FP_ID'] = FP
    metrics_exp['FN_ID'] = FN
    metrics_exp['CCR'] = correct_classification_rate(tn, fn, tp)

    df_metrics = pd.DataFrame.from_dict(metrics_exp,
                                        orient='index',
                                        columns=['Metrics, Results'])
    df_metrics.to_csv(path_output/f'all_images_metrics_selina.csv')
    print(df_metrics)
