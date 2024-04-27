import numpy as np
import os
import nibabel as nib
from pathlib import Path

from matplotlib import pyplot as plt

from src.mesoECE.data_structure.patient import Patient
from src.mesoECE.methods import AbstractMethod
from src.mesoECE.methods.utils import define_masks_volume, setup_directories

from src.mesoECE.methods.superspels.utils import (plot_curves,
                                                  save_superspels_masks,
                                                  save_curves_and_interp_to_csv)
from src.mesoECE.methods.classifier.utils import (superspels_labels_with_ece,
                                                  define_ece_curves,
                                                  define_ece_mask)
from src.mesoECE.methods.superspels.utils import define_mean_intensity_curves


class ECE(AbstractMethod):
    def __init__(self, path: Path, ref_t: int, domain: str = None):
        super().__init__()

        self.thread_safe = False
        self.predicted_diagnosis = []
        self.path_ece = path
        self.ref_t = ref_t
        self.domain = domain

    def apply(self, patient: Patient, **kwargs):
        try:
            print("\r", end='')
            print("ECE processing......", end="", flush=True)

            # Paths
            setup_directories(self.path_ece, ['ece_images',
                                              'benign_images',
                                              'plots',
                                              'curves_df'])
            mask = patient.get_image(self.ref_t).masks[
                "supervoxels"].data.astype(np.int32)
            mean_intensity, _ = define_mean_intensity_curves(patient=patient,
                                                             mask=mask,
                                                             domain=self.domain)
            ece_labels, ece_mask = self.process_ece_detection(patient,
                                                              mean_intensity)

            if ece_labels and self.significant_volume(ece_mask, patient):
                self.process_malignant_case(patient, ece_labels, ece_mask,
                                            mean_intensity)
            else:
                self.process_benign_case(patient, ece_labels, ece_mask)

            patient.path_masks['ece'] = self.path_ece
        except Exception as e:
            print(e)
            print("Error in id: ", patient.id)
        new_patient = Patient(path=patient.path,
                              path_masks=patient.path_masks,
                              id=patient.id,
                              diagnosis=patient.diagnosis,
                              subclass_diagnosis=patient.subclass_diagnosis,
                              nodular=patient.nodular)
        return new_patient

    def result(self):
        return self.predicted_diagnosis

    def process_ece_detection(self, patient, mean_intensity):
        index_270 = patient.time_points.index(270)
        ece_labels = superspels_labels_with_ece(index_270=index_270,
                                                mean_intensity_curve=mean_intensity)
        ece_mask = define_ece_mask(ece_labels,
                                   patient.get_image(self.ref_t).masks[
                                       "supervoxels"].data.astype(np.int32))
        return ece_labels, ece_mask

    def significant_volume(self, ece_mask, patient):
        pleural_vol = define_masks_volume(
            patient.get_image(self.ref_t).masks["pleural_region"].data.astype(
                np.int32))
        ece_vol = define_masks_volume(ece_mask)
        return len(ece_mask) > 0 and ece_vol > pleural_vol * 0.0001

    def process_malignant_case(self, patient, ece_labels, ece_mask,
                               mean_intensity):
        self.predicted_diagnosis.append(
            [patient.id, patient.diagnosis, 1, patient.subclass_diagnosis,
             patient.nodular, len(ece_labels), define_masks_volume(ece_mask)])

        ece_curves, benign_curves = define_ece_curves(
            mean_intensity_curves=mean_intensity, ece_labels=ece_labels)
        save_curves_and_interp_to_csv(patient, ece_curves,
                                      self.path_ece / 'curves_df', 'ece')
        save_curves_and_interp_to_csv(patient, benign_curves,
                                      self.path_ece / 'curves_df', 'benign')
        plot_curves(curve=ece_curves, time_points=patient.time_points,
                    mask=ece_mask,
                    filename=str(self.path_ece / 'plots' / f'ece_{patient.id}'),
                    mean_plot=True, selected_labels=ece_labels)

        nifti_args = patient.get_image(self.ref_t).nifti_props
        nib.save(nib.Nifti1Image(ece_mask, **nifti_args),
                 str(self.path_ece / 'ece_images' / patient.get_image(
                     self.ref_t).filename))

    def process_benign_case(self, patient, ece_labels, ece_mask):
        self.predicted_diagnosis.append(
            [patient.id, patient.diagnosis, 0, patient.subclass_diagnosis,
             patient.nodular, len(ece_labels), define_masks_volume(ece_mask)])
