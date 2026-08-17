import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from meso_regions.mesoECE.experiment import Experiment
from meso_regions.run.config import define_config_dilate

data_root = Path("data")

path_masks = Path("/data_lids/home/taylla/PycharmProjects/meso/data/resample"
                  "/masks")
path_classes = Path("/data_lids/home/taylla/PycharmProjects/meso/data"
                    "/classes_subclasses_nodular.csv")
path_images = Path("/data_lids/home/taylla/PycharmProjects/meso/data/resample"
                   "/images_orig_reg")
path_output = Path("/data_lids/home/taylla/PycharmProjects/meso/output")

ids = list(sorted((pd.read_csv(path_classes)["ID"]).to_list()))
threads = min(os.cpu_count(), len(ids))

for dilation_radius in tqdm([2, 3, 4]):
    for p_center_distance in tqdm(
            [0, 0.1, 0.5, 0.9]):
        for otsu in [False, True]:
            # Define configuration

            config_dilation = define_config_dilate(
                ref_t=270,
                dilation_radius=dilation_radius,
                p_center_distance=p_center_distance,
                otsu=otsu,
                mask_to_dilate='fluid')
            config = [config_dilation]
            if otsu:
                experiment_name = (f"fluid_d_{dilation_radius}_"
                                   f"p_{p_center_distance}_otsu")
            else:
                experiment_name = (f"fluid_d_{dilation_radius}_"
                                   f"p_{p_center_distance}")

            experiment = Experiment(path_masks=path_masks, ids=ids,
                                    path_classes=path_classes,
                                    path_images=path_images,
                                    path_experiments=path_output / "HyperparameterTuning/Dilation",
                                    experiment_name=experiment_name,
                                    config=config,
                                    threads=threads)
            experiment.execute_pipeline()
