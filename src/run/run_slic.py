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
                   "/HyperparameterTuning/exp")

path_masks_dilated = Path("/data_lids/home/taylla/PycharmProjects/meso/output/"
                          "/HyperparameterTuning/exp")
list_masks_dilated = os.listdir(path_masks_dilated)

path_splits = Path("/data_lids/home/taylla/PycharmProjects/meso/data/splits")
mode = 'train'

if mode == 'train':
    ids = list(sorted(
        (pd.read_csv(path_splits / "train_set_class_4.csv")["ID"]).to_list()))
else:
    ids = list(sorted(
        (pd.read_csv(path_splits / "test_set_class_4.csv")["ID"]).to_list()))
threads = min(os.cpu_count(), len(ids))

n_segments = np.arange(0, 800, 50).tolist()
p_seeds = [0.001, 0.01]
for mask in tqdm(list_masks_dilated, desc="Masks"):
    path_masks = path_masks_dilated / mask / 'Dilate'
    metrics_mask = {}

    for n_segment in tqdm(range(0, 800, 50), desc="N_Segments"):
        for compactness in tqdm([0.1, 1, 10], desc="Compactness"):
            if n_segment == 0:
                for p_seeds_final in tqdm(p_seeds, desc="P_final_Seeds"):
                    # Define configuration
                    config_slic = define_config_slic(
                        ref_t=270,
                        n_segments=n_segment,
                        compactness=compactness,
                        p_seeds_final=p_seeds_final)
                    config_superspels = define_config_superspels(ref_t=270)
                    config_ece = define_config_ece(ref_t=270)
                    config = [config_slic, config_superspels, config_ece]
                    experiment_name = (f"{n_segment}_"
                                       f"{compactness}_{p_seeds_final}")

                    experiment = Experiment(
                        path_masks=path_masks,
                        ids=ids,
                        path_classes=path_classes,
                        path_images=path_images,
                        path_experiments=path_output / mask / 'SLIC' / 'split_class' / mode,
                        experiment_name=experiment_name,
                        config=config,
                        threads=threads)
                    exp = experiment.execute_pipeline()
                    metrics_exp = experiment.classifier_metrics()
                    print(metrics_exp)

                    metrics_mask[
                        f'{n_segment}_{compactness}_{p_seeds_final}'] = metrics_exp
            else:

                # Define configuration
                config_slic = define_config_slic(
                    ref_t=270,
                    n_segments=n_segment,
                    compactness=compactness,
                    p_seeds_final=0)
                config_superspels = define_config_superspels(ref_t=270)
                config_ece = define_config_ece(ref_t=270)
                config = [config_slic, config_superspels, config_ece]
                experiment_name = (f"{n_segment}_"
                                   f"{compactness}_{0}")

                experiment = Experiment(
                    path_masks=path_masks,
                    ids=ids,
                    path_classes=path_classes,
                    path_images=path_images,
                    path_experiments=path_output / mask / 'SLIC' / 'split_class' / mode,
                    experiment_name=experiment_name,
                    config=config,
                    threads=threads)
                exp = experiment.execute_pipeline()
                metrics_exp = experiment.classifier_metrics()
                print(metrics_exp)
                metrics_mask[
                    f'{n_segment}_{compactness}_{0}'] = metrics_exp

    df_metrics_mask = pd.DataFrame(metrics_mask)
    df_metrics_mask.to_csv(
        path_output / mask / 'SLIC' / 'split_class' / mode / f"metrics_{mask}_slic_{mode}.csv")

