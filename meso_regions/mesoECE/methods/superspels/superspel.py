from pathlib import Path
from meso_regions.mesoECE.data_structure import Patient
from meso_regions.mesoECE.methods import AbstractMethod
from meso_regions.mesoECE.methods.superspels.utils import (

    save_and_plot_curves, calculate_curves)

from meso_regions.mesoECE.methods.utils import setup_directories


class Superspel(AbstractMethod):
    def __init__(self, path: Path, ref_t: int):
        super().__init__()
        self.path_superspels = path
        self.ref_t = ref_t
        self.thread_safe = False

    def apply(self, patient: Patient, **kwargs):
        try:
            print("\r", end='')
            print("Superspels processing......", end="", flush=True)

            # Paths
            setup_directories(self.path_superspels,
                              ['plots', 'curves_df', 'plots/mean', 'plots/std'])

            mean_intensity, std_intensity = calculate_curves(patient)

            save_and_plot_curves(path=self.path_superspels,
                                 patient=patient,
                                 curves=mean_intensity,
                                 curve_name='mean_intensity',
                                 mask=patient.get_image(self.ref_t).masks[
                                     'supervoxels'].data)
            save_and_plot_curves(path=self.path_superspels,
                                 patient=patient,
                                 curves=std_intensity,
                                 curve_name='std_intensity',
                                 mask=patient.get_image(self.ref_t).masks[
                                     'supervoxels'].data)
        except Exception as e:
            print(e)
            print(f'Error with {patient.id}')
        new_patient = Patient(path=patient.path, path_masks=patient.path_masks,
                              id=patient.id, diagnosis=patient.diagnosis,
                              subclass_diagnosis=patient.subclass_diagnosis,
                              nodular=patient.nodular)
        return new_patient

    def result(self):
        return None
