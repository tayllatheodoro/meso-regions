import nibabel as nib
from pathlib import Path
import numpy as np

from meso_regions.mesoECE.data_structure.patient import Patient
from meso_regions.mesoECE.methods import AbstractMethod


class SubLung(AbstractMethod):
    def __init__(self, path: Path, ref_t: int = 270):
        super().__init__()
        self.path_mask_sub_lung = path
        self.ref_t = ref_t
        self.thread_safe = True

    def apply(self, patient: Patient, **kwargs):
        try:
            nifti_args = patient.get_image(self.ref_t).get_mask(
                'fluid').nifti_props
            mask_dilated = patient.get_image(self.ref_t).get_mask(
                'pleural_region').data
            mask_lung = patient.get_image(self.ref_t).get_mask('lung').data

            mask_add = np.logical_xor(mask_dilated, mask_lung).astype(np.int32)
            patient.path_masks['pleural_region'] = self.path_mask_sub_lung
            nib.save(nib.Nifti1Image(mask_add, **nifti_args),
                     str(self.path_mask_sub_lung / patient.get_image(
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
