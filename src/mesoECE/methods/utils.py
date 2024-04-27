import os
from pathlib import Path

import numpy as np
from src.mesoECE.data_structure import Patient


def setup_directories(path: Path, dir_names: list):
    for directory in dir_names:
        os.makedirs(path / directory, exist_ok=True)
def define_masks_volume(mask):
    mask_volume = np.count_nonzero(mask)
    if mask_volume is None:
        return 0
    else:
        return mask_volume


def correct_image_background(patient: Patient, t):
    image = patient.get_image(t).data
    image_corrected = image - patient.background_otsu(t)
    return image_corrected


def correct_images_background(patient: Patient):
    images_corrected = []
    for t in patient.time_points:
        images_corrected = correct_image_background(patient, t)
    return images_corrected



