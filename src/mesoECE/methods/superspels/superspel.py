import os
from pathlib import Path

import numpy as np

from src.mesoECE.data_structure import Patient, MRImage
from src.mesoECE.methods import AbstractMethod
from src.mesoECE.methods.utils import correct_images_background
from src.mesoECE.methods.superspels.utils import (define_superspels_mask,
                                                  define_mean_intensity_curves,
                                                  save_curves_and_interp_to_csv,
                                                  plot_curves)
import nibabel as nib


class Superspel(AbstractMethod):
    def __init__(self, path: Path, ref_t: int, domain: str = 'REG'):
        super().__init__()
        self.path_superspels = path
        self.ref_t = ref_t
        self.domain = domain
        self.thread_safe = False

    def apply(self, patient: Patient, **kwargs):
        try:
            print("\r", end='')
            print("Superspels processing......", end="", flush=True)

            # Paths
            path_ss_df = self.path_superspels / 'curves_df'
            path_plot = self.path_superspels / 'plots'
            os.makedirs(path_ss_df, exist_ok=True)
            os.makedirs(path_plot, exist_ok=True)

            # Superspels mask (4D array) and nifti_args
            # patient.ss_mask, patient.nifti_args = define_superspels_mask(
            #     patient=patient,
            #     domain=self.domain,
            #     ref_t=self.ref_t)

            patient.nifti_args = patient.get_image(self.ref_t).nifti_props
            patient.ss_mask = patient.get_image(self.ref_t).masks[
                "supervoxels"].data.astype(np.int32)

            # Correct images background
            # images_corrected = correct_images_background(patient=patient)
            #
            # for t in patient.time_points:
            #     # Save corrected images
            #     nifti_args = patient.get_image(t).nifti_props
            #     nib.save(nib.Nifti1Image(images_corrected[t].data,
            #                              **nifti_args),
            #              str(self.path_superspels / patient.get_image(
            #                  t).filename))

            # Define mean intensity curves for all superspels
            mean_intensity, std_intensity = define_mean_intensity_curves(
                patient=patient,
                mask=patient.get_image(self.ref_t).masks[
                    "supervoxels"].data.astype(np.int32),
                domain=self.domain)

            # # Save mean intensity curves to csv and interpolate of it
            save_curves_and_interp_to_csv(
                patient=patient,
                curves=mean_intensity,
                path=path_ss_df,
                curve_name='mean_intensity_curves')

            # # Save mean intensity curves to csv and interpolate of it
            save_curves_and_interp_to_csv(
                patient=patient,
                curves=std_intensity,
                path=path_ss_df,
                curve_name='std_intensity_curves')
            os.makedirs(path_plot / 'mean', exist_ok=True)
            os.makedirs(path_plot / 'std', exist_ok=True)
            # # Plot mean intensity curves
            plot_curves(curve=mean_intensity,
                        time_points=patient.time_points,
                        mask=patient.get_image(self.ref_t).masks[
                            'supervoxels'].data,
                        filename=str(path_plot /'mean' /f'{patient.id}'),
                        mean_plot=True,
                        title='All Curves')

            plot_curves(curve=std_intensity,
                        time_points=patient.time_points,
                        mask=patient.get_image(self.ref_t).masks[
                            'supervoxels'].data,
                        filename=str(path_plot /'std'/ f'{patient.id}'),
                        mean_plot=True,
                        title='All Curves')

        except Exception as e:
            print(e)
            print(f'Error with {patient.id}')
        new_patient = Patient(path=patient.path, path_masks=patient.path_masks,
                              id=patient.id, diagnosis=patient.diagnosis,
                              subclass_diagnosis=patient.subclass_diagnosis,
                              nodular=patient.nodular)
        return new_patient
