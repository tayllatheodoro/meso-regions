import os
from pathlib import Path

import pandas as pd

from src.mesoECE.experiment import Experiment
from src.run.config import define_config_dilation

data_root = Path("../data")
path_patients = data_root / ("/app/data/src/data/experiments/Resample"
                             "/Image_Orig_Reg/Resample/00001/images")

path_masks = data_root / ("/app/data/fluid_mask")
path_classes = data_root / "classes_subclasses_nodular.csv"
path_output = data_root / f"dilation"

ids = list(sorted(pd.read_csv(path_classes)["ID"].tolist()))
threads = min(os.cpu_count(), len(ids))

for dilation_radius in [2, 3, 4, 5]:
    for p_center_distance in [0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        for otsu in [True, False]:
            # Define configuration

            config_dilation = define_config_dilation(
                ref_t=270,
                dilation_radius=dilation_radius,
                p_center_distance=0.5,
                otsu=otsu,
                mask_to_dilate='fluid')
            config = [config_dilation]
            experiment_name = (f"fluid_{dilation_radius}/"
                               f"{p_center_distance}_{otsu}")

            experiment = Experiment(path_masks=path_masks, ids=ids,
                                    path_classes=path_classes,
                                    path_images=path_patients,
                                    path_experiments=data_root / "experiments",
                                    experiment_name=experiment_name,
                                    config=config,
                                    threads=threads)
