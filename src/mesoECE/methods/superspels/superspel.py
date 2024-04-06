from pathlib import Path
from src.mesoECE.data_structure import Patient, MRImage
from src.mesoECE.methods import AbstractMethod
from src.mesoECE.methods.utils import correct_images_background
from src.mesoECE.methods.superspels.utils import (define_superspels_mask,
                                                  define_superspels_curves,
                                                  save_curves_and_interp_to_csv,
                                                  plot_curves)


class Superspel(AbstractMethod):
    def __init__(self, path: Path, ref_t: int, domain: str = 'REG'):
        super().__init__()
        self.path_superspels = path
        self.ref_t = ref_t
        self.domain = domain
        self.thread_safe = True

    def apply(self, patient: Patient, **kwargs):
        try:
            path_ss_df = self.path_superspels / 'superspels_df'
            path_plot = self.path_superspels / 'plots'

            # Superspels mask (4D array) and nifti_args
            patient.ss_mask, patient.nifti_args = define_superspels_mask(
                patient=patient,
                domain=self.domain,
                ref_t=self.ref_t)
            # Correct images background
            images_corrected = correct_images_background(patient=patient)

            # Define mean intensity curves for all superspels and save to csv
            patient.curves['mean_intensity'] = define_superspels_curves(
                patient=patient,
                images_corrected=images_corrected,
                ss_mask=patient.ss_mask,
                domain=self.domain,
                ref_t=self.ref_t)

            save_curves_and_interp_to_csv(
                patient=patient,
                curves=patient.curves['mean_intensity'],
                ref_t=self.ref_t,
                path=path_ss_df,
                curve_name='mean_intensity_curves')

            plot_curves(curve=patient.curves['mean_intensity'],
                        time_points=patient.time_points,
                        mask=patient.get_image(self.ref_t).masks[
                            'supervoxels'].data,
                        filename=str(path_plot / f'{
                        MRImage.resolve_name(
                            patient.id, self.ref_t,
                            ".png")}'),
                        mean_plot=True,
                        title='All Curves')

        except:
            print(f'Error with {patient.id}')
        new_patient = Patient(path=patient.path, path_masks=patient.path_masks,
                              id=patient.id, diagnosis=patient.diagnosis,
                              subclass_diagnosis=patient.subclass_diagnosis,
                              nodular=patient.nodular)
        return new_patient
