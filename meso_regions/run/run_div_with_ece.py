import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from meso_regions.mesoECE.experiment import Experiment
from meso_regions.run.config import define_config_div_with_ece

path_classes = Path("/data_lids/home/taylla/PycharmProjects/meso/data"
                    "/classes_subclasses_nodular.csv")
path_images = Path("/data_lids/home/taylla/PycharmProjects/meso/data/resample"
                   "/images_orig_reg")
path_output = Path("/data_lids/home/taylla/PycharmProjects/meso/output"
                   "/HyperparameterTuning/No_P_masks/teste_code_div_ece")

path_masks_dilated = Path("/data_lids/home/taylla/PycharmProjects/meso/output/"
                          "/HyperparameterTuning/Dilation")

path_splits = Path("/data_lids/home/taylla/PycharmProjects/meso/data/splits")

list_masks_dilated = os.listdir(path_masks_dilated)
ids_train = [12, 93]
# ids_train = list(sorted(
#     (pd.read_csv(path_splits / "training_set_classes_4.csv")["ID"]).to_list()))
# ids_test = list(sorted(
#     (pd.read_csv(path_splits / "test_set_classes_4.csv")["ID"]).to_list()))
threads = min(os.cpu_count(), len(ids_train))

metrics_all = {}

predict_only_small = [True, False]

# = [mask for mask in list_masks_dilated if mask not in done]
for mask in tqdm(list_masks_dilated, desc="Masks"):

    path_masks = path_masks_dilated / mask / 'Dilate'
    metrics_mask = {}
    for n_segment in tqdm([2, 3, 5, 10, 15, 20], desc="N_Segments"):
        for compactness in tqdm([0.1, 1, 10], desc="Compactness"):
            for p_size in tqdm([500,1000,2000, 5000, 10000], desc="P_size"):
                for p_o_s in predict_only_small:
                    # Define configuration

                    config_div_ece = define_config_div_with_ece(
                        n_segments=n_segment,
                        compactness=compactness,
                        p_size=p_size,
                        predict_only_small=p_o_s)
                    config = [config_div_ece]

                    experiment_name = (f"{mask}/{n_segment}_"
                                       f"{compactness}_{p_size}_{p_o_s}")

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
                    # metrics_exp = experiment.classifier_metrics()
                    # print(metrics_exp)
                    #
                    # metrics_all[
                    #     f'{mask}_{n_segment}_{compactness}_{p_size}_{p_o_s}'] = metrics_exp
                    # metrics_mask[
                    #     f'{n_segment}_{compactness}_{p_size}_{p_o_s}'] = metrics_exp

    df_metrics_mask = pd.DataFrame(metrics_mask)
    df_metrics_mask.to_csv(path_output / f"metrics_{mask}_exp.csv")

df_metrics = pd.DataFrame(metrics_all)
df_metrics.to_csv(path_output / f"metrics_all_exp.csv")
