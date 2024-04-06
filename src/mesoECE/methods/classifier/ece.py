import numpy as np
import os
from pathlib import Path
from src.mesoECE.data_structure.patient import Patient, MRImage
from src.mesoECE.methods import AbstractMethod
from src.mesoECE.methods.utils import (define_masks_volume,
                                       correct_images_background)
from src.mesoECE.methods.utils_ece import (ece_label_selection,
                                           plot_ece_curves,
                                           define_superspels_mask,
                                           define_superspels_curves,
                                           save_superspels_masks,
                                           save_curves_and_interp_to_csv,
                                           define_ece_curves, define_ece_mask)


class ECE(AbstractMethod):
    def __init__(self, path: Path, ref_t: int, domain: str = None):
        super().__init__()

        self.thread_safe = False
        self.predicted_diagnosis = []
        self.path_ece = path
        self.ref_t = ref_t
        self.domain = domain

    def apply(self, patient: Patient, **kwargs):
        path_m_images = self.path_ece / 'ece_images'
        path_b_images = self.path_ece / 'benign_images'
        path_plot = self.path_ece / 'plots'
        path_ss_df = self.path_ece / 'superspels_df'

        os.makedirs(path_ss_df, exist_ok=True)
        os.makedirs(path_plot, exist_ok=True)
        os.makedirs(path_m_images, exist_ok=True)
        os.makedirs(path_b_images, exist_ok=True)

        try:
            print("\r", end='')
            print("ECE processing......", end="", flush=True)

            ss_mask, nifti_args = define_superspels_mask(patient=patient,
                                                         domain=self.domain,
                                                         ref_t=self.ref_t)
            images_corrected = correct_images_background(patient=patient)

            ss_mean_curves = define_superspels_curves(
                patient=patient,
                images_corrected=images_corrected,
                ss_mask=ss_mask,
                domain=self.domain,
                ref_t=self.ref_t)
            save_curves_and_interp_to_csv(patient=patient,
                                          curves=ss_mean_curves,
                                          ref_t=self.ref_t,
                                          path=path_ss_df,
                                          curve_name='all')

            index_270 = patient.time_points.index(270)

            ece_labels = ece_label_selection(
                index_270=index_270,
                ss_mean=ss_mean_curves)

            ece_mask, benign_mask = define_ece_mask(ece_labels, ss_mask)

            # Calculate the volume of the pleural mask
            pleural_mask = patient.get_image(self.ref_t).masks[
                "pleural_region"].data.astype(np.int32)
            pleural_vol = define_masks_volume(mask=pleural_mask)

            # Calculate the volume of the ECE mask
            ece_mask_vol = 0
            if self.domain == 'REG':
                ece_mask_vol = ece_mask[-1]

            elif self.domain == 'ORIG':
                ece_mask_vol = ece_mask[index_270]
            ece_vol = define_masks_volume(mask=ece_mask_vol)

            if ece_labels and ece_vol > pleural_vol * 0.0001:
                self.predicted_diagnosis.append(
                    [patient.id, patient.diagnosis, 1,
                     patient.subclass_diagnosis, patient.nodular,
                     ece_labels.__len__(), ece_vol])

                ss_ece_curves = define_ece_curves(
                    len_time_points=len(patient.time_points),
                    ss_mean_curves=ss_mean_curves,
                    ece_labels=ece_labels)

                save_superspels_masks(ss_mask=ece_mask,
                                      nifti_args=nifti_args,
                                      patient=patient,
                                      domain=self.domain,
                                      ref_t=self.ref_t,
                                      path=path_m_images)
                ece_mask_plot = None
                if self.domain == 'REG':
                    ece_mask_plot = ece_mask[-1]

                elif self.domain == 'ORIG':
                    ece_mask_plot = ece_mask[index_270]
                plot_ece_curves(curves=ss_mean_curves,
                                time_points=patient.time_points,
                                ece_mask=ece_mask_plot,
                                filename=str(path_plot /
                                             MRImage.resolve_name(
                                                 patient.id, self.ref_t,
                                                 "png")))

                save_curves_and_interp_to_csv(patient=patient,
                                              curves=ss_ece_curves,
                                              ref_t=self.ref_t,
                                              path=path_ss_df,
                                              curve_name='ece')

            else:
                self.predicted_diagnosis.append(
                    [patient.id, patient.diagnosis, 0,
                     patient.subclass_diagnosis, patient.nodular,
                     ece_labels.__len__(), ece_vol])

            save_superspels_masks(ss_mask=benign_mask,
                                  nifti_args=nifti_args,
                                  patient=patient,
                                  domain=self.domain,
                                  ref_t=self.ref_t,
                                  path=path_b_images)

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
