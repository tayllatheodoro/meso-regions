import numpy as np
import nibabel as nib
from pathlib import Path
from src.mesoECE.data_structure.patient import Patient
from src.mesoECE.methods import AbstractMethod
from src.mesoECE.methods.utils import define_masks_volume, setup_directories, \
    save_nii_mask

from src.mesoECE.methods.superspels.utils import (plot_curves,
                                                  save_curves_and_interp_to_csv,
                                                  save_and_plot_curves,
                                                  calculate_curves)
from src.mesoECE.methods.classifier.utils import (superspels_labels_with_ece,
                                                  define_ece_curves,
                                                  define_ece_mask)
from src.mesoECE.methods.superspels.utils import define_mean_std_intensity_curves


class ECE(AbstractMethod):
    def __init__(self, path: Path, ref_t: int):
        super().__init__()

        self.thread_safe = False
        self.predicted_diagnosis = []
        self.path_ece = path
        self.ref_t = ref_t

    def apply(self, patient: Patient, **kwargs):
        try:
            print("\r", end='')
            print("ECE processing......", end="", flush=True)

            # Paths
            setup_directories(self.path_ece, ['ece_images',
                                              'benign_images',
                                              'plots',
                                              'curves_df'])

            mean_intensity, std_intensity = calculate_curves(patient=patient)
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
        ece_labels = superspels_labels_with_ece(index_270=patient.time_points.index(270),
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

        save_and_plot_curves(path=self.path_ece,
                             patient=patient,
                             curves=ece_curves,
                             curve_name='ece',
                             mask=ece_mask)
        # save_and_plot_curves(path=self.path_ece,
        #                      patient=patient,
        #                      curves=benign_curves,
        #                      curve_name='benign',
        #                      mask=ece_mask)

        save_nii_mask(patient=patient,
                      path=self.path_ece / 'ece_images',
                      mask=ece_mask)

    def process_benign_case(self, patient, ece_labels, ece_mask):
        self.predicted_diagnosis.append(
            [patient.id, patient.diagnosis, 0, patient.subclass_diagnosis,
             patient.nodular, len(ece_labels), define_masks_volume(ece_mask)])
