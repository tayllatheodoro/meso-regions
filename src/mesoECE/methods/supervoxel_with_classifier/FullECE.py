import numpy as np
import os
import pandas as pd
import nibabel as nib
import skimage.measure
import scipy
from pathlib import Path

from src.mesoECE.data_structure.patient import Patient, Diagnosis
from src.mesoECE.methods import AbstractMethod


class FullECE(AbstractMethod):
    def __init__(self, path: Path, ref_t: int, filter_size: int = 5):
        super().__init__()
        self.thread_safe = True
        self.predicted_diagnosis = []
        self.path_ece = path
        self.ref_t = ref_t
        self.filter_size = filter_size

    def apply(self, patient: Patient, **kwargs):

        nifti_args = patient.get_image(self.ref_t).nifti_props

        try:

            images = []
            maximum_vec = np.zeros(patient.get_image(self.ref_t).data.shape)
            ece = np.ones_like(maximum_vec)
            max_index = patient.time_points.index(270)
            mask = patient.get_image(self.ref_t).masks[
                "pleural_region"].data.astype(np.int32)

            for t in patient.time_points:
                # image mean filter using skitmage
                img_otsu = skimage.filters.threshold_otsu(
                    patient.get_image(t).data)
                img_filtered = scipy.ndimage.uniform_filter(
                    patient.get_image(t).data - img_otsu, size=self.filter_size)
                img_filtered[mask == 0] = 0
                images.append(img_filtered)

            for i, t in enumerate(patient.time_points):
                if i < max_index:
                    ece = np.logical_and(ece, images[i] < images[i + 1])
                elif i > max_index:
                    ece = np.logical_and(ece, images[i] < images[i - 1])

            ece = ece.astype(np.int32)
            ece_sum = np.sum(ece)
            volume_mask = np.sum(mask)
            if ece_sum > 0.0001 * volume_mask:
                self.predicted_diagnosis.append(
                    [patient.id, patient.diagnosis, Diagnosis.MALIGNANT,
                     ece_sum, ece_sum])
                # print(
                #     f"[{patient.id}, {patient.diagnosis}, {Diagnosis.MALIGNANT}, {ece_sum}]")

                path_images = self.path_ece / 'images'
                os.makedirs(path_images, exist_ok=True)
                nib.save(nib.Nifti1Image(ece, **nifti_args),
                         str(path_images / patient.get_image(
                             self.ref_t).filename))
            else:
                self.predicted_diagnosis.append(
                    [patient.id, patient.diagnosis, Diagnosis.NON_MALIGNANT,
                     ece_sum, ece_sum])
                # print(
                #     f"[{patient.id}, {patient.diagnosis}, {Diagnosis.NON_MALIGNANT}, {ece_sum}]")

            patient.path_masks['ece'] = self.path_ece
        except:
            print("Error in id: ", patient.id)
        new_patient = Patient(path=patient.path,
                              path_masks=patient.path_masks,
                              id=patient.id,
                              diagnosis=patient.diagnosis,
                              subclass_diagnosis=patient.subclass_diagnosis,
                              nodular=patient.nodular)
        return new_patient

    def results(self):
        return self.predicted_diagnosis
