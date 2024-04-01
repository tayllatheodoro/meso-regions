import os
import numpy as np
from pathlib import Path

from src.mesoECE.data_structure import Patient
from src.mesoECE.methods import AbstractMethod


class SICLE(AbstractMethod):
    def __init__(self, path: Path, ref_t: int, n_init: int, n_final: int,
                 p_seeds_init: float = 0.1, p_seeds_final: float = 0.01,
                 conn_opt: str = 'fmax',crit_opt: str = 'minsc',
                 pen_opt: str = 'none'):
        super().__init__()
        self.path_supervoxels = path
        self.ref_t = ref_t
        self.n_init = n_init
        self.n_final = n_final
        self.thread_safe = True
        self.p_seeds_init = p_seeds_init
        self.p_seeds_final = p_seeds_final
        self.seeds = []
        self.conn_opt = conn_opt
        self.crit_opt = crit_opt
        self.pen_opt = pen_opt

    def apply(self, patient: Patient, **kwargs):
        try:
            # Get number of seeds
            n_seeds_init = self.n_init
            n_seeds_final = self.n_final

            if self.n_init == 0 or self.n_final == 0:
                mask = patient.get_image(self.ref_t).get_mask(
                    "pleural_region").data
                volume = int(np.sum(mask))
                if self.n_init == 0 and self.n_final == 0:
                    n_seeds_init = int(volume * self.p_seeds_init)
                    n_seeds_final = int(volume * self.p_seeds_final)
                elif self.n_init == 0 and self.n_final > 0:
                    n_seeds_init = int(volume * self.p_seeds_init)
                elif self.n_init > 0 and self.n_final == 0:
                    n_seeds_final = int(volume * self.p_seeds_final)

            self.seeds.append([patient.id, n_seeds_init, n_seeds_final])

            if n_seeds_init > n_seeds_final:
                # Input and output paths to call DISF
                path_img = patient.get_image(
                    self.ref_t).path / patient.get_image(
                    self.ref_t).filename
                path_mask = patient.get_image(self.ref_t).path_masks[
                                'pleural_region'] / patient.get_image(
                    self.ref_t).filename
                path_out_mask = self.path_supervoxels / patient.get_image(
                    self.ref_t).filename

                # Apply SICLE in Current Patient in Volume from reference image
                # and save supervoxels mask
                cmd = (f"/app/data/ift/bin/RunSICLE --img {path_img} --n0{n_seeds_init}"
                       f" --nf {n_seeds_final} --out {path_out_mask} --mask {path_mask}"
                       f" --conn-opt {self.conn_opt} --crit-opt {self.crit_opt}"
                       f" --pen-opt {self.pen_opt}")
                # --conn-opt      IFT connectivity function. Options: fmax, fsum, custom. Default: fmax
                # --crit-opt      Seed removal criterion. Options: size, minsc, maxsc, spread, custom. Default: minsc
                # --pen-opt       Seed relevance penalization. Options: none, obj, bord, osb, bobs, custom. Default: none

                if os.system(cmd) == -1:
                    print(
                        "Error at iftDISF. Please compile the program iftDISF")
                    exit(-1)
            else:
                raise Exception("N Seeds Init is >= N seed Final, it muss "
                                "be >>")

            # Update patient path masks
            patient.path_masks['supervoxels'] = self.path_supervoxels
        except:
            print(f'Error with {patient.id}')
        new_patient = Patient(path=patient.path, path_masks=patient.path_masks,
                              id=patient.id, diagnosis=patient.diagnosis,
                              subclass_diagnosis=patient.subclass_diagnosis,
                              nodular=patient.nodular)
        return new_patient

    def result(self):
        return self.seeds
