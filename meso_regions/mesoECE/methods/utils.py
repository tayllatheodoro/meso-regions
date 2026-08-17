import os
from pathlib import Path

import numpy as np
from meso_regions.mesoECE.data_structure import Patient
import nibabel as nib


def setup_directories(path: Path, dir_names: list):
    for directory in dir_names:
        os.makedirs(path / directory, exist_ok=True)


def define_masks_volume(mask: np.ndarray):
    mask_volume = np.count_nonzero(mask)
    return mask_volume


def save_nii_mask(path: Path, patient: Patient, mask: np.ndarray, ref_t=270):
    img = patient.get_image(ref_t)
    nifti_args = img.nifti_props
    nib.save(nib.Nifti1Image(mask, **nifti_args),
             str(path / img.filename))


def correct_image_background(patient: Patient, t):
    image = patient.get_image(t).data
    image_corrected = image - patient.background_otsu(t)
    return image_corrected


def correct_images_background(patient: Patient):
    images_corrected = []
    for t in patient.time_points:
        images_corrected = correct_image_background(patient, t)
    return images_corrected
