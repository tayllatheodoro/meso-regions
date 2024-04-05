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


def define_ece_curves(patient: Patient, ece_mask):
    ece_mean_curve = np.zeros(ece_mask.max() + 1,
                              patient.time_points.__len__())

    rps = skimage.measure.regionprops(ece_mask)
    for rp in rps:
        slice_bbox = tuple(
            [slice(dim_start, dim_finish) for dim_start, dim_finish in
             zip(rp.bbox[:3], rp.bbox[3:])])
        lbl_in_bbox = rp.image

        for t in patient.time_points:
            img = correct_image_background(patient, t)
            img_in_bbox = img[slice_bbox]
            ece_mean_curve[rp.label - 1, patient.time_points.index(t)] = \
                img_in_bbox[lbl_in_bbox > 0]
    return ece_mean_curve
