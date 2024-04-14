import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.mesoECE.experiment import Experiment
from src.run.config import define_config_slic, define_config_ece, \
    define_config_superspels,define_config_full_ece

path_classes = Path("/data_lids/home/taylla/PycharmProjects/meso/data"
                    "/classes_subclasses_nodular.csv")
path_images = Path("/data_lids/home/taylla/PycharmProjects/meso/data/resample"
                   "/images_orig_reg")
path_output = Path("/data_lids/home/taylla/PycharmProjects/meso/output"
                   "/HyperparameterTuning/FULL_ECE")

path_masks_dilated = Path("/data_lids/home/taylla/PycharmProjects/meso/output/"
                          "/HyperparameterTuning/Dilation")

path_splits = Path("/data_lids/home/taylla/PycharmProjects/meso/data/splits")

list_masks_dilated = os.listdir(path_masks_dilated)

ids_train = list(sorted(
    (pd.read_csv(path_splits / "training_set_classes_4.csv")["ID"]).to_list()))
# ids_test = list(sorted(
#     (pd.read_csv(path_splits / "test_set_classes_4.csv")["ID"]).to_list()))
threads = min(os.cpu_count(), len(ids_train))

metrics_all = {}

filter_size = np.arange(1, 20, 1).tolist()

with_mask = [True, False]
for f in filter_size:
    for w_m  in with_mask:
        if w_m:
            for mask in tqdm(list_masks_dilated, desc="Masks"):
                path_masks = path_masks_dilated / mask / 'Dilate'
                metrics_mask = {}

                # Define configuration

                config_slic = define_config_full_ece(
                                        ref_t=270,
                                        filter_size=filter_size,
                                        with_mask = True )
                                    # config_superspels = define_config_superspels(ref_t=270,
                                    #                                              domain='REG')
                config_ece = define_config_ece(ref_t=270, domain='REG')
                config = [config_slic, config_ece]

                experiment_name = f"{mask}/{filter_size}"

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
                f'{mask}_{filter_size}'] = metrics_exp
                metrics_mask[f'{filter_size}'] = metrics_exp

                df_metrics_mask = pd.DataFrame(metrics_mask)
                df_metrics_mask.to_csv(path_output / f"metrics_{mask}_exp.csv")
        else:

            metrics_no_mask = {}
            config_slic = define_config_full_ece(
                ref_t=270,
                filter_size=filter_size,
                with_mask=True)
            config_ece = define_config_ece(ref_t=270, domain='REG')
            config = [config_slic, config_ece]

            experiment_name = f"no_mask/{filter_size}"

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
                f'no_mask_{filter_size}'] = metrics_exp
            metrics_no_mask[f'{filter_size}'] = metrics_exp

            df_metrics_mask = pd.DataFrame(metrics_mask)
            df_metrics_mask.to_csv(path_output / f"metrics_no_mask_{filter_size}_exp.csv")

df_metrics = pd.DataFrame(metrics_all)
df_metrics.to_csv(path_output / f"metrics_all_exp.csv")


def test():
    ids_train = [11, 12, 90]

    # ids_train = list(sorted(
    #      (pd.read_csv(path_splits / "training_set_classes_4.csv")["ID"]).to_list()))
    # ids_test = list(sorted(
    #     (pd.read_csv(path_splits / "test_set_classes_4.csv")["ID"]).to_list()))
    threads = min(os.cpu_count(), len(ids_train))
    n_segment = 0
    compactness = 0.1
    p_seeds_final = 0.011

    path_masks = Path(
        '/data_lids/home/taylla/PycharmProjects/meso/output/HyperparameterTuning/Dilation/fluid_2_0.7_False/Dilate/00001')
    mask = 'fluid_2_0.7_False'
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
