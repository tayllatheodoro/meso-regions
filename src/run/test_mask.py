import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.mesoECE.experiment import Experiment
from src.run.config import define_config_slic, define_config_ece, \
    define_config_superspels

path_classes = Path("/data_lids/home/taylla/PycharmProjects/meso/data"
                    "/classes_subclasses_nodular_ST.csv")
path_images = Path("/data_lids/home/taylla/PycharmProjects/meso/data/resample"
                   "/images_orig_reg")
path_output = Path("/data_lids/home/taylla/PycharmProjects/meso/output"
                   "/HyperparameterTuning/teste_mask")

path_masks_dilated = Path("/data_lids/home/taylla/PycharmProjects/meso/output/"
                          "/HyperparameterTuning/Dilation")

path_splits = Path("/data_lids/home/taylla/PycharmProjects/meso/data/splits")

list_masks_dilated = os.listdir(path_masks_dilated)
ids_train =[11,14,90]
# ids_train = list(sorted(
#     (pd.read_csv(path_splits / "training_set_classes_4.csv")["ID"]).to_list()))
# ids_test = list(sorted(
#     (pd.read_csv(path_splits / "test_set_classes_4.csv")["ID"]).to_list()))
threads = min(os.cpu_count(), len(ids_train))
path_masks = path_masks_dilated / 'fluid_d_2_p_0' / 'Dilate'



# Define configuration
config_slic = define_config_slic(
                        ref_t=270,
                        n_segments=10,
                        compactness=1,
                        p_seeds_final=0)
config_superspels = define_config_superspels(ref_t=270,
                                              domain='REG')
config_ece = define_config_ece(ref_t=270,
                                                   domain='REG')
config = [config_slic,config_superspels, config_ece]
experiment_name = (f"teste_mask")

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
