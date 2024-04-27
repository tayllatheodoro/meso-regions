import nibabel as nib
import numpy as np
from skimage.segmentation import slic
from pathlib import Path

from src.mesoECE.data_structure import Patient
from src.mesoECE.methods import AbstractMethod

from filelock import Timeout, FileLock

from src.mesoECE.methods.utils import define_masks_volume


class SLIC(AbstractMethod):
    def __init__(self, path: Path, ref_t: int, n_segments: int,
                 compactness: float, p_seeds_final: float = 0.01):
        super().__init__()
        self.path_supervoxels = path
        self.ref_t = ref_t
        self.n_segments = n_segments
        self.compactness = compactness
        self.thread_safe = True
        self.p_seeds_final = p_seeds_final
        self.seeds = []

    def apply(self, patient: Patient, **kwargs):

        try:
            mask = patient.get_image(self.ref_t).masks["pleural_region"].data
            n_seeds_final = self.calculate_seeds(patient, mask)

            # Get Spacing
            image = patient.get_image(self.ref_t).data
            nifti_args = patient.get_image(self.ref_t).nifti_props
            image_header = nifti_args["header"]
            mri_spacing = image_header.get_zooms()

            # Apply SLIC in Current Patient in Volume from reference image

            supervoxels_mask = slic(image, mask=mask, channel_axis=None,
                                    compactness=self.compactness,
                                    n_segments=n_seeds_final,
                                    spacing=mri_spacing,
                                    start_label=1)
            # Save supervoxels mask
            nib.save(nib.Nifti1Image(supervoxels_mask.astype(np.int32),
                                     **nifti_args),
                     str(self.path_supervoxels / patient.get_image(
                         self.ref_t).filename))

            # Update patient path masks
            patient.path_masks['supervoxels'] = self.path_supervoxels
        except Exception as e:
            print("Error in id: ", patient.id)
            print(e)
        new_patient = Patient(path=patient.path, path_masks=patient.path_masks,
                              id=patient.id,
                              diagnosis=patient.diagnosis,
                              subclass_diagnosis=patient.subclass_diagnosis,
                              nodular=patient.nodular)
        return new_patient

    def result(self):
        return self.seeds

    def calculate_seeds(self, patient: Patient, mask: np.ndarray):
        # Get number of seeds
        n_seeds_final = self.n_segments
        if self.n_segments == 0:
            volume = define_masks_volume(mask)
            n_seeds_final = int(volume * self.p_seeds_final)
        self.seeds.append([patient.id, 0, n_seeds_final])
        return n_seeds_final
