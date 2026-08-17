import nibabel as nib
from pathlib import Path
import numpy as np

from meso_regions.mesoECE.data_structure.patient import Patient
from meso_regions.mesoECE.methods import AbstractMethod


class AddLung(AbstractMethod):
    def __init__(self, path: Path, ref_t: int =270):
        super().__init__()
        self.path_fluid_and_lung = path
        self.ref_t = ref_t
        self.thread_safe = True

    def apply(self, patient: Patient, **kwargs):
        try:
            nifti_args = patient.get_image(self.ref_t).get_mask(
                'fluid').nifti_props
            mask_fluid = patient.get_image(self.ref_t).get_mask(
                'fluid').data
            mask_lung = patient.get_image(self.ref_t).get_mask('lung').data

            mask_add = np.maximum(mask_fluid, mask_lung)
            patient.path_masks['fluid_and_lung'] = self.path_fluid_and_lung
            nib.save(nib.Nifti1Image(mask_add, **nifti_args),
                     str(self.path_fluid_and_lung / patient.get_image(
                         self.ref_t).filename))


        except:
            print("Error in id: ", patient.id)

        new_patient = Patient(path=patient.path,
                              path_masks=patient.path_masks,
                              id=patient.id,
                              diagnosis=patient.diagnosis,
                              subclass_diagnosis=patient.subclass_diagnosis,
                              nodular=patient.nodular)
        return new_patient
