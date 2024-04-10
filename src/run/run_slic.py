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
                   "/HyperparameterTuning/SLIC")

path_masks_dilated = Path("/data_lids/home/taylla/PycharmProjects/meso/output/"
                          "/HyperparameterTuning/Dilation")

path_splits = Path("/data_lids/home/taylla/PycharmProjects/meso/data/splits")

list_masks_dilated = os.listdir(path_masks_dilated)

#ids_train = [89,254,443]

ids_train = list(sorted(
     (pd.read_csv(path_splits / "training_set_classes_4.csv")["ID"]).to_list()))
# ids_test = list(sorted(
#     (pd.read_csv(path_splits / "test_set_classes_4.csv")["ID"]).to_list()))
threads = min(os.cpu_count(), len(ids_train))

metrics_all = {}
sentinel_metrics = ['ACC', 'AUC', 'F1']
n_segments = np.arange(0, 1050, 50).tolist()
p_seeds = np.arange(0.001, 0.05, 0.001).tolist()
for mask in list_masks_dilated:
    path_masks = path_masks_dilated / mask / 'Dilate/00001'

    for n_segment in range(0, 1050, 50):
        for compactness in tqdm([0.1, 1, 10, 100]):
            if n_segment == 0:
                for p_seeds_final in p_seeds:
                    # Define configuration
                    config_slic = define_config_slic(
                        ref_t=270,
                        n_segments=n_segment,
                        compactness=compactness,
                        p_seeds_final=p_seeds_final)
                    config_superspels = define_config_superspels(ref_t=270,
                                                                 domain='REG')
                    config_ece = define_config_ece(ref_t=270,
                                                   domain='REG')
                    config = [config_slic, config_superspels, config_ece]
                    experiment_name = (f"{mask}/{n_segment}_"
                                       f"{compactness}_{p_seeds_final}")

                    experiment = Experiment(
                        path_masks=path_masks,
                        ids=ids_train,
                        path_classes=path_classes,
                        path_images=path_images,
                        path_experiments=path_output / "train",
                        experiment_name=experiment_name,
                        config=config,
                        threads=threads)
                    exp = experiment.execute_pipeline()
                    metrics_exp = experiment.classifier_metrics()
                    print(metrics_exp)

                    metrics_all[
                        f'{mask}_{n_segment}_{compactness}_{p_seeds_final}'] = metrics_exp
            else:

                # Define configuration
                config_slic = define_config_slic(
                    ref_t=270,
                    n_segments=n_segment,
                    compactness=compactness,
                    p_seeds_final=0)
                config_superspels = define_config_superspels(ref_t=270,
                                                             domain='REG')
                config_ece = define_config_ece(ref_t=270,
                                               domain='REG')
                config = [config_slic, config_ece]
                experiment_name = (f"{mask}/{n_segment}_"
                                   f"{compactness}_{0}")

                experiment = Experiment(
                    path_masks=path_masks,
                    ids=ids_train,
                    path_classes=path_classes,
                    path_images=path_images,
                    path_experiments=path_output / "train",
                    experiment_name=experiment_name,
                    config=config,
                    threads=threads)
                exp = experiment.execute_pipeline()
                metrics_exp = exp.classifier_metrics()
                print(metrics_exp)

                metrics_all[
                    f'{mask}_{n_segment}_{compactness}_{0}'] = metrics_exp

df_metrics = pd.DataFrame(metrics_all)
df_metrics.to_csv(path_output / f"metrics_all_exp.csv")
