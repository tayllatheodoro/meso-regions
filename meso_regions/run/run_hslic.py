import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# add ../../ to the path

from meso_regions.mesoECE.experiment import Experiment
from meso_regions.run.config import define_config_ece, \
    define_config_superspels, define_config_hslic

path_classes = Path("/data_lids/home/taylla/PycharmProjects/meso/data"
                    "/classes_subclasses_nodular.csv")
path_images = Path("/data_lids/home/taylla/PycharmProjects/meso/data/resample"
                   "/images_orig_reg")
path_output = Path("/data_lids/home/taylla/PycharmProjects/meso/output"
                   "/exp_dilatation_P")

path_masks_dilated = Path("/data_lids/home/taylla/PycharmProjects/meso/data"
                          "/orig/dilation_patient")

list_masks_dilated = os.listdir(path_masks_dilated)

path_splits = Path("/data_lids/home/taylla/PycharmProjects/meso/data/splits")

for s in os.listdir(path_splits):
    s_n = s.split('_')[-1].split('.')[0]
    mode = s.split('_')[0]
    ids = list(sorted(
        (pd.read_csv(path_splits / s)["ID"]).to_list()))
    threads = min(os.cpu_count()-5, len(ids))

    n_segments = np.arange(0, 800, 50).tolist()
    p_seeds = [0.0001, 0.0003, 0.0005, 0.0007, 0.001, 0.003, 0.005, 0.007, 0.01,
               0.03, 0.05, 0.07, 0.1]
    if True:
        mask = 'dilated_patient'
        path_masks = path_masks_dilated
        metrics_mask = {}
        for compactness in tqdm([0.1, 1, 10], desc="Compactness"):
            for n_segment in tqdm(range(0, 800, 50), desc="N_Segments"):
                for n_segments_h in tqdm(range(2, 6, 1), desc="N_Segments_H"):

                    if n_segment == 0:
                        for p_seeds_final in tqdm(p_seeds,
                                                  desc="P_final_Seeds"):
                            # Define configuration
                            config_hslic = define_config_hslic(
                                ref_t=270,
                                n_segments=n_segment,
                                n_segments_h=n_segments_h,
                                compactness=compactness,
                                p_seeds_final=p_seeds_final)
                            config_superspels = define_config_superspels(
                                ref_t=270)
                            config_ece = define_config_ece(ref_t=270)
                            config = [config_hslic, config_superspels,
                                      config_ece]
                            experiment_name = f"{compactness}/{p_seeds_final}/{n_segments_h}"

                            experiment = Experiment(
                                path_masks=path_masks,
                                ids=ids,
                                path_classes=path_classes,
                                path_images=path_images,
                                path_experiments=path_output / mask / 'HSLIC' / f'split_{s_n}' / mode,
                                experiment_name=experiment_name,
                                config=config,
                                threads=threads)
                            exp = experiment.execute_pipeline()
                            metrics_exp = experiment.classifier_metrics()
                            print(metrics_exp)

                            metrics_mask[
                                f'{compactness}_{p_seeds_final}_{n_segments_h}'] = metrics_exp
                    else:

                        # Define configuration
                        config_hslic = define_config_hslic(
                            ref_t=270,
                            n_segments=n_segment,
                            n_segments_h=n_segments_h,
                            compactness=compactness,
                            p_seeds_final=0)
                        config_superspels = define_config_superspels(ref_t=270)
                        config_ece = define_config_ece(ref_t=270)
                        config = [config_hslic, config_superspels, config_ece]
                        experiment_name = f"{compactness}/{n_segment}/{n_segments_h}"

                        experiment = Experiment(
                            path_masks=path_masks,
                            ids=ids,
                            path_classes=path_classes,
                            path_images=path_images,
                            path_experiments=path_output / mask / 'HSLIC' / f'split_{s_n}' / mode,
                            experiment_name=experiment_name,
                            config=config,
                            threads=threads)
                        exp = experiment.execute_pipeline()
                        metrics_exp = experiment.classifier_metrics()
                        print(metrics_exp)
                        metrics_mask[
                            f'{compactness}_{n_segment}_{n_segments_h}'] = metrics_exp

        df_metrics_mask = pd.DataFrame(metrics_mask)
        df_metrics_mask.to_csv(
            path_output / mask / 'HSLIC' / f'split_{s_n}' / mode / f"metrics_{mask}_hslic_{mode}.csv")
