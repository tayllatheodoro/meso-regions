import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from src.mesoECE.experiment import Experiment
from src.run.config import define_config_full_ece

path_classes = Path("/data_lids/home/taylla/PycharmProjects/meso/data"
                    "/classes_subclasses_nodular.csv")
path_images = Path("/data_lids/home/taylla/PycharmProjects/meso/data/resample"
                   "/images_orig_reg")
path_output = Path("/data_lids/home/taylla/PycharmProjects/meso/output"
                   "/HyperparameterTuning/FULL_ECE")

path_masks_dilated = Path("/data_lids/home/taylla/PycharmProjects/meso/output"
                          "/HyperparameterTuning/Dilation")

path_splits = Path("/data_lids/home/taylla/PycharmProjects/meso/data/splits")

list_masks_dilated = os.listdir(path_masks_dilated)
# ids_train =[11,12,90]
ids_train = list(sorted(
    (pd.read_csv(path_splits / "training_set_classes_4.csv")["ID"]).to_list()))
# ids_test = list(sorted(
#     (pd.read_csv(path_splits / "test_set_classes_4.csv")["ID"]).to_list()))
threads = min(os.cpu_count(), len(ids_train))

metrics_all = {}

filter_size = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
               19, 20]

with_mask = [True, False]
for mask in tqdm(list_masks_dilated, desc="Masks"):
    metrics_mask = {}
    for f in filter_size:
        path_masks = path_masks_dilated / mask / 'Dilate'


        # Define configuration

        config_full_ece = define_config_full_ece(
            ref_t=270,
            filter_size=f,
            with_mask=True)
        #config_superspels = define_config_superspels(ref_t=270,
        #                                              domain='REG')
        config = [config_full_ece]

        experiment_name = f"{mask}/{f}"

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
            f'{mask}_{f}'] = metrics_exp
        metrics_mask[f'{f}'] = metrics_exp

    df_metrics_mask = pd.DataFrame(metrics_mask)
    df_metrics_mask.to_csv(path_output / f"metrics_{mask}_exp.csv")

metrics_no_mask = {}
for f in filter_size:
    path_masks = path_masks_dilated / 'fluid_d_2_p_0' / 'Dilate'

    config_slic = define_config_full_ece(
        ref_t=270,
        filter_size=f,
        with_mask=False)
    config = [config_slic]

    experiment_name = f"no_mask/{f}"

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
        f'no_mask_{f}'] = metrics_exp
    metrics_no_mask[f'{f}'] = metrics_exp

    df_metrics_mask = pd.DataFrame(metrics_no_mask)
    df_metrics_mask.to_csv(
        path_output / f"metrics_no_mask_{f}_exp.csv")

# df_metrics = pd.DataFrame(metrics_all)
# df_metrics.to_csv(path_output / f"metrics_all_exp.csv")
