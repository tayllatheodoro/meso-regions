import numpy as np
import os
from pathlib import Path
from src.mesoECE.data_structure.patient import Patient, MRImage
from src.mesoECE.methods import AbstractMethod
from src.mesoECE.methods.utils import define_masks_volume

from src.mesoECE.methods.superspels.utils import (plot_curves,
                                                  save_superspels_masks,
                                                  save_curves_and_interp_to_csv)
from src.mesoECE.methods.classifier.utils import (superspels_labels_with_ece,
                                                  define_ece_curves,
                                                  define_ece_mask)


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
            path_m_images = self.path_ece / 'ece_images'
            path_b_images = self.path_ece / 'benign_images'
            path_plot = self.path_ece / 'plots'
            path_ss_df = self.path_ece / 'superspels_df'

            os.makedirs(path_ss_df, exist_ok=True)
            os.makedirs(path_plot, exist_ok=True)
            os.makedirs(path_m_images, exist_ok=True)
            os.makedirs(path_b_images, exist_ok=True)
            print("\r", end='')
            print("ECE processing......", end="", flush=True)

            # select which superspels have ece pattern
            index_270 = patient.time_points.index(270)
            ece_labels = superspels_labels_with_ece(
                index_270=index_270,
                mean_intensity_curve=patient.curves['mean_intensity'])
            # Define mask with superspels with ece pattern
            # and benign masks
            ece_mask, benign_mask = define_ece_mask(ece_labels, patient.ss_mask)

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

            # if there are superspels with ece pattern and the volume of the
            # ece mask is greater than 0.01% of the pleural mask volume to
            # reduce false positives

            if ece_labels and ece_vol > pleural_vol * 0.0001:
                self.predicted_diagnosis.append(
                    [patient.id, patient.diagnosis, 1,
                     patient.subclass_diagnosis, patient.nodular,
                     ece_labels.__len__(), ece_vol])

                # Define mean intensity curves for superspels with ece pattern
                ece_curves, benign_curves = define_ece_curves(
                    len_time_points=len(patient.time_points),
                    mean_intensity_curves=patient.curves['mean_intensity'],
                    ece_labels=ece_labels)

                patient.curves['ece'] = ece_curves
                patient.curves['benign'] = benign_curves

                save_curves_and_interp_to_csv(patient=patient,
                                              curves=ece_curves,
                                              ref_t=self.ref_t,
                                              path=path_ss_df,
                                              curve_name='ece')

                save_curves_and_interp_to_csv(patient=patient,
                                              curves=benign_curves,
                                              ref_t=self.ref_t,
                                              path=path_ss_df,
                                              curve_name='benign')
                # Plot ece curves vs time
                ece_mask_plot = None
                benign_mask_plot = None
                if self.domain == 'REG':
                    ece_mask_plot = ece_mask[-1]
                    benign_mask_plot = benign_mask[-1]

                elif self.domain == 'ORIG':
                    ece_mask_plot = ece_mask[index_270]
                    benign_mask_plot = benign_mask[index_270]
                plot_curves(curve=benign_curves,
                            time_points=patient.time_points,
                            mask=ece_mask_plot,
                            filename=str(path_plot / f'ece_{
                            MRImage.resolve_name(
                                patient.id, self.ref_t,
                                ".png")}'),
                            mean_plot=True)

                plot_curves(curve=benign_curves,
                            time_points=patient.time_points,
                            mask=benign_mask_plot,
                            filename=str(path_plot / f'benign_{
                            MRImage.resolve_name(
                                patient.id, self.ref_t,
                                ".png")}'),
                            mean_plot=True,
                            title='Benign Curves')

                # Save ece mask
                save_superspels_masks(ss_mask=ece_mask,
                                      nifti_args=patient.nifti_args,
                                      patient=patient,
                                      domain=self.domain,
                                      ref_t=self.ref_t,
                                      path=path_m_images)

            else:
                self.predicted_diagnosis.append(
                    [patient.id, patient.diagnosis, 0,
                     patient.subclass_diagnosis, patient.nodular,
                     ece_labels.__len__(), ece_vol])

            # Save benign mask
            save_superspels_masks(ss_mask=benign_mask,
                                  nifti_args=patient.nifti_args,
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
