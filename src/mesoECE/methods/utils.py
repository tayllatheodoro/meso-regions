import numpy as np
import pandas as pd
import skimage
from matplotlib import pyplot as plt

from src.mesoECE.data_structure import Patient


def define_masks_volume(mask):
    mask_temp = np.where(mask, 1, 0)
    mask_volume = np.sum(mask_temp)

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



