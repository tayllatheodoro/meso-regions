import os
from pathlib import Path

import pandas as pd

from src.mesoECE.experiment import Experiment
from src.run.config import define_config_slic, define_config_ece, \
    define_config_superspels, define_config_disf, define_config_full_ece

path_classes = Path("/data_lids/home/taylla/PycharmProjects/meso/data"
                    "/classes_subclasses_nodular.csv")
path_images = Path("/data_lids/home/taylla/PycharmProjects/meso/data/resample"
                   "/images_orig_reg")
path_masks_dilated = Path("/data_lids/home/taylla/PycharmProjects/meso/output/"
                          "/HyperparameterTuning/Dilation")
path_output = Path("/data_lids/home/taylla/PycharmProjects/meso/output"
                   "/HyperparameterTuning/test")
path_train_metrics = Path("/data_lids/home/taylla/PycharmProjects/meso/output"
                          "/HyperparameterTuning/train_metrics")
path_splits = Path("/data_lids/home/taylla/PycharmProjects/meso/data/splits")
ids_test = list(sorted(
    (pd.read_csv(path_splits / "test_set_classes_4.csv")["ID"]).to_list()))
threads = min(os.cpu_count(), len(ids_test))
methods = ['slic', 'disf', 'fullece']


for method in methods:
    df = pd.read_csv(path_train_metrics / f"best_metrics_{method}.csv")
    auc_column = df['AUC']
    acc_column = df['ACC']
    f1_column = df['F1']

    auc_parameter = auc_column[1]
    acc_parameter = acc_column[1]
    f1_parameter = f1_column[1]
    print(method)
    print(df)
    # SLIC: f'{mask}_{n_segment}_{compactness}_{p_seeds_final}']
    # disf: f"{mask}_{n_final}_{p_seeds_final}")
    # fullece: {mask}_{f}
    metrics = {}
    for metric in ['AUC', 'ACC', 'F1']:
        if metric == 'AUC':
            p = auc_parameter
        elif metric == 'ACC':
            p = acc_parameter
        else:
            p = f1_parameter
        dilatation_radius = p.split('_')[2]
        p_center = p.split('_')[4]
        if p.split('_')[5] == 'otsu':
            otsu = 'True'
            mask = f'fluid_d_{dilatation_radius}_p_{p_center}_otsu'
            if method == 'slic':

                n_segment = p.split('_')[6]
                compactness = p.split('_')[7]
                p_seeds_final = p.split('_')[8]
            elif method == 'disf':
                n_final = p.split('_')[6]
                p_seeds_final = p.split('_')[7]
            elif method == 'fullece':
                f = p.split('_')[6]
        else:
            mask = f'fluid_d_{dilatation_radius}_p_{p_center}'
            if method == 'slic':
                n_segment = p.split('_')[5]
                compactness = p.split('_')[6]
                p_seeds_final = p.split('_')[7]
            elif method == 'disf':
                n_final = p.split('_')[5]
                p_seeds_final = p.split('_')[6]
            elif method == 'fullece':
                f = p.split('_')[5]

        path_masks = path_masks_dilated / mask / 'Dilate'
        if method == 'slic':
            config_slic = define_config_slic(
                ref_t=270,
                n_segments=int(n_segment),
                compactness=float(compactness),
                p_seeds_final=float(p_seeds_final))
            config_superspels = define_config_superspels(ref_t=270,
                                                         domain='REG')
            config_ece = define_config_ece(ref_t=270, domain='REG')
            config = [config_slic, config_superspels, config_ece]
        elif method == 'disf':
            config_disf = define_config_disf(
                ref_t=270,
                n_init=int(n_final) * 10,
                n_final=int(n_final),
                p_seeds_init=float(p_seeds_final) * 10,
                p_seeds_final=float(p_seeds_final))
            config_ece = define_config_ece(ref_t=270, domain='REG')
            config = [config_disf, config_ece]
        elif method == 'fullece':
            config_full_ece = define_config_full_ece(
                ref_t=270,
                filter_size=int(f),
                with_mask=True)
            config = [config_full_ece]

        experiment_name = (f"{method}/{metric}")

        experiment = Experiment(
            path_masks=path_masks,
            ids=ids_test,
            path_classes=path_classes,
            path_images=path_images,
            path_experiments=path_output,
            experiment_name=experiment_name,
            config=config,
            threads=threads)
        exp = experiment.execute_pipeline()
        metrics_exp = experiment.classifier_metrics()
        print(metrics_exp)

        metrics[metric] = metrics_exp
