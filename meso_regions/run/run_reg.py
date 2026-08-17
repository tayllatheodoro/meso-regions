import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from meso_regions.mesoECE.experiment import Experiment
from meso_regions.run.config import define_ants_reg

path_masks = Path("/data_lids/home/taylla/PycharmProjects/meso/data/resample"
                  "/masks")
path_classes = Path("/data_lids/home/taylla/PycharmProjects/meso/data"
                    "/classes_subclasses_nodular.csv")
path_images = Path("/data_lids/home/taylla/PycharmProjects/meso/data/resample"
                   "/images_orig_reg")
path_output = Path("/data_lids/home/taylla/PycharmProjects/meso/output")

ids = list(sorted((pd.read_csv(path_classes)["ID"]).to_list()))
threads = min(os.cpu_count(), len(ids))

config_reg = define_ants_reg(ref_t=270)
config = [config_reg]
experiment_name = "ants_reg"
experiment = Experiment(path_masks=path_masks, ids=ids,
                        path_classes=path_classes,
                        path_images=path_images,
                        path_experiments=path_output,
                        experiment_name=experiment_name,
                        config=config,
                        threads=threads)
experiment.execute_pipeline()
