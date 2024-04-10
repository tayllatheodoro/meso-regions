import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.mesoECE.experiment import Experiment
from src.run.config import define_config_slic, define_config_ece, \
    define_config_superspels, define_config_disf

path_classes = Path("/data_lids/home/taylla/PycharmProjects/meso/data"
                    "/classes_subclasses_nodular.csv")
path_images = Path("/data_lids/home/taylla/PycharmProjects/meso/data/resample"
                   "/images_orig_reg")
path_output = Path("/data_lids/home/taylla/PycharmProjects/meso/output"
                   "/HyperparameterTuning/DISF")

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


p_seeds = [0.001, 0.005, 0.01, 0.05]
for mask in tqdm(list_masks_dilated, desc="Masks"):
    path_masks = path_masks_dilated / mask / 'Dilate/00001'
    metrics_mask = {}

    for n_final in tqdm(range(0, 1050, 50), desc="N_Segments"):

            if n_final == 0:
                for p_seeds_final in tqdm(p_seeds, desc="P_final_Seeds"):
                    # Define configuration
                    config_disf = define_config_disf(
                        ref_t = 270,
                        n_init = n_final*10,
                        n_final = n_final,
                        p_seeds_init =p_seeds_final*10,
                        p_seeds_final=p_seeds_final)
                    config_superspels = define_config_superspels(ref_t=270,
                                                                 domain='REG')
                    config_ece = define_config_ece(ref_t=270,
                                                   domain='REG')
                    config = [config_disf, config_superspels, config_ece]
                    experiment_name = (f"{mask}/{n_final}_"
                                       f"{p_seeds_final}")

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
                        f'{mask}_{n_final}_{p_seeds_final}'] = metrics_exp
                    metrics_mask[
                        f'{mask}_{n_final}_{p_seeds_final}'] = metrics_exp
            else:

                # Define configuration
                config_disf = define_config_disf(
                    ref_t=270,
                    n_init=n_final * 10,
                    n_final=n_final,
                    p_seeds_init=0,
                    p_seeds_final=0)
                config_superspels = define_config_superspels(ref_t=270,
                                                             domain='REG')
                config_ece = define_config_ece(ref_t=270,
                                               domain='REG')
                config = [config_disf, config_ece]
                experiment_name = (f"{mask}/{n_final}_"
                                       f"{0}")

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
                    f'{mask}_{n_final}_{0}'] = metrics_exp
                metrics_mask[
                    f'{n_final}_{0}'] = metrics_exp

    df_metrics_mask = pd.DataFrame(metrics_mask)
    df_metrics_mask.to_csv(path_output / f"metrics_{mask}_exp.csv")

df_metrics = pd.DataFrame(metrics_all)
df_metrics.to_csv(path_output / f"metrics_all_exp.csv")


# def test():
#     ids_train = [11, 12, 90]
#
#     # ids_train = list(sorted(
#     #      (pd.read_csv(path_splits / "training_set_classes_4.csv")["ID"]).to_list()))
#     # ids_test = list(sorted(
#     #     (pd.read_csv(path_splits / "test_set_classes_4.csv")["ID"]).to_list()))
#     threads = min(os.cpu_count(), len(ids_train))
#     n_final = 0
#     compactness = 0.1
#     p_seeds_final = 0.011
#
#     path_masks = Path(
#         '/data_lids/home/taylla/PycharmProjects/meso/output/HyperparameterTuning/Dilation/fluid_2_0.7_False/Dilate/00001')
#     mask = 'fluid_2_0.7_False'
#     config_slic = define_config_fi(
#         ref_t=270,
#         n_segments=n_final,
#         compactness=compactness,
#         p_seeds_final=p_seeds_final)
#     config_superspels = define_config_superspels(ref_t=270,
#                                                  domain='REG')
#     config_ece = define_config_ece(ref_t=270,
#                                    domain='REG')
#     config = [config_slic, config_superspels, config_ece]
#     experiment_name = (f"{mask}/{n_segment}_"
#                        f"{compactness}_{p_seeds_final}")
#
#     experiment = Experiment(
#         path_masks=path_masks,
#         ids=ids_train,
#         path_classes=path_classes,
#         path_images=path_images,
#         path_experiments=path_output / "train",
#         experiment_name=experiment_name,
#         config=config,
#         threads=threads)
#     exp = experiment.execute_pipeline()
#     metrics_exp = experiment.classifier_metrics()
#     print(metrics_exp)
