import os
from pathlib import Path
from src.mesoECE.data_structure import Patient
from src.mesoECE.methods import AbstractMethod
from src.mesoECE.methods.utils import define_masks_volume


class DISF(AbstractMethod):
    def __init__(self, path: Path, ref_t: int, n_init: int, n_final: int,
                 p_seeds_init: float = 0.1, p_seeds_final: float = 0.01,
                 ift_path: str = '/data_lids/home/taylla/ift'):
        super().__init__()
        self.path_supervoxels = path
        self.ref_t = ref_t
        self.n_init = n_init
        self.n_final = n_final
        self.thread_safe = True
        self.p_seeds_init = p_seeds_init
        self.p_seeds_final = p_seeds_final
        self.seeds = []
        self.ift_path = ift_path

    def apply(self, patient: Patient, **kwargs):
        try:
            n_seeds_init, n_seeds_final = self.calculate_seeds(patient)

            self.execute_disf(patient, n_seeds_init, n_seeds_final)

            # Update patient path masks
            patient.path_masks['supervoxels'] = self.path_supervoxels

        except Exception as e:
            print(f'Error with patient {patient.id}: {str(e)}')
        new_patient = Patient(path=patient.path, path_masks=patient.path_masks,
                              id=patient.id, diagnosis=patient.diagnosis,
                              subclass_diagnosis=patient.subclass_diagnosis,
                              nodular=patient.nodular)
        return new_patient

    def result(self):
        return self.seeds

    def execute_disf(self, patient: Patient, n_seeds_init: int,
                     n_seeds_final: int):
        """Execute the DISF command using subprocess for better control and
        security."""
        img = patient.get_image(self.ref_t)
        filename = img.filename
        path_img = img.path / filename
        path_mask = img.path_masks['pleural_region'] / filename
        path_out_mask = self.path_supervoxels / filename

        cmd = (f"{self.ift_path}/bin/iftDISF {path_img} "
               f"{n_seeds_init} {n_seeds_final} {path_out_mask} {path_mask}")

        if os.system(cmd) == -1:
            print(
                "Error at iftDISF. Please compile the program iftDISF")
            exit(-1)

    def calculate_seeds(self, patient: Patient):
        """Calculate initial and final seeds based on volume."""
        n_seeds_init = self.n_init
        n_seeds_final = self.n_final

        if self.n_init == 0 and self.n_final == 0:
            mask = patient.get_image(self.ref_t).get_mask(
                "pleural_region").data
            volume = define_masks_volume(mask)
            n_seeds_init = int(volume * self.p_seeds_init)
            n_seeds_final = int(volume * self.p_seeds_final)

        self.seeds.append([patient.id, n_seeds_init, n_seeds_final])
        if n_seeds_init <= n_seeds_final:
            raise ValueError("Initial seeds must be less than final seeds.")

        return n_seeds_init, n_seeds_final
