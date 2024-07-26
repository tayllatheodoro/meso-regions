import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.mesoECE.experiment import Experiment
from src.run.config import define_config_ece, \
    define_config_superspels, define_config_disf

path_classes = Path("/data_lids/home/taylla/PycharmProjects/meso/data"
                    "/classes_subclasses_nodular.csv")
path_images = Path("/data_lids/home/taylla/PycharmProjects/meso/data/resample"
                   "/images_orig_reg")
path_output = Path("/data_lids/home/taylla/PycharmProjects/meso/output"
                   "/exp_final")

path_masks_dilated = Path(
    "/data_lids/home/taylla/PycharmProjects/meso/data/orig/dilation_patient")

path_splits = Path("/data_lids/home/taylla/PycharmProjects/meso/data/splits")

list_masks_dilated = os.listdir(path_masks_dilated)

for s in os.listdir(path_splits):
    s_n = s.split('_')[-1].split('.')[0]
    if s_n == '1':
        mode = s.split('_')[0]

        ids = list(sorted(
            (pd.read_csv(path_splits / s)["ID"]).to_list()))
        threads = min(os.cpu_count(), len(ids))

        p_seeds = [0.0001, 0.0003, 0.0005, 0.0007, 0.001, 0.003, 0.005, 0.007,
                   0.01, 0.03, 0.05, 0.07, 0.1]

        mask = 'dilated_patient'
        path_masks = path_masks_dilated
        metrics_mask = {}

        for n_final in tqdm(range(0, 800, 50), desc="N_Segments"):

            if n_final == 0:
                metrics_mask_p ={}
                for p_seeds_final in tqdm(p_seeds, desc="P_final_Seeds"):
                    # Define configuration
                    config_disf = define_config_disf(
                        ref_t=270,
                        n_init=n_final * 10,
                        n_final=n_final,
                        p_seeds_init=p_seeds_final * 10,
                        p_seeds_final=p_seeds_final,
                        ift_path='/data_lids/home/taylla/ift')
                    config_superspels = define_config_superspels(ref_t=270)
                    config_ece = define_config_ece(ref_t=270)
                    config = [config_disf, config_superspels, config_ece]
                    experiment_name = f"{p_seeds_final}"

                    experiment = Experiment(
                        path_masks=path_masks,
                        ids=ids,
                        path_classes=path_classes,
                        path_images=path_images,
                        path_experiments=path_output / mask / 'DISF' / f'split_{s_n}' / mode,
                        experiment_name=experiment_name,
                        config=config,
                        threads=threads)
                    exp = experiment.execute_pipeline()
                    metrics_exp = experiment.classifier_metrics()
                    print(metrics_exp)

                    metrics_mask_p[
                        f'{p_seeds_final}'] = metrics_exp
                df_metrics_mask = pd.DataFrame(metrics_mask_p)
                df_metrics_mask.to_csv(
                    path_output / mask / 'DISF' / f'split_{s_n}' / mode / f"metrics_{mask}_disf_{mode}_percentage.csv")
            else:
                # Define configuration
                config_disf = define_config_disf(
                    ref_t=270,
                    n_init=n_final * 10,
                    n_final=n_final,
                    p_seeds_init=0,
                    p_seeds_final=0,
                    ift_path='/data_lids/home/taylla/ift')
                config_superspels = define_config_superspels(ref_t=270)
                config_ece = define_config_ece(ref_t=270)
                config = [config_disf, config_superspels, config_ece]
                experiment_name = f"{n_final}"

                experiment = Experiment(
                    path_masks=path_masks,
                    ids=ids,
                    path_classes=path_classes,
                    path_images=path_images,
                    path_experiments=path_output / mask / 'DISF' / f'split_{s_n}' / mode,
                    experiment_name=experiment_name,
                    config=config,
                    threads=threads)
                exp = experiment.execute_pipeline()
                metrics_exp = experiment.classifier_metrics()
                print(metrics_exp)

                metrics_mask[f'{n_final}'] = metrics_exp

            df_metrics_mask = pd.DataFrame(metrics_mask)
            df_metrics_mask.to_csv(
                path_output / mask / 'DISF' / f'split_{s_n}' / mode / f"metrics_{mask}_disf_{mode}_fixed.csv")
