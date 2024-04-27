import os
from pathlib import Path

import numpy as np
from src.mesoECE.data_structure import Patient


def setup_directories(path: Path, dir_names: list):
    for directory in dir_names:
        os.makedirs(path / directory, exist_ok=True)


def define_masks_volume(mask):
    mask_volume = np.count_nonzero(mask)
    return mask_volume
