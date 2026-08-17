import nibabel as nib
import numpy as np
import skimage
from skimage.segmentation import slic
from pathlib import Path

from meso_regions.mesoECE.data_structure import Patient
from meso_regions.mesoECE.methods import AbstractMethod

from filelock import Timeout, FileLock

from meso_regions.mesoECE.methods.utils import define_masks_volume, save_nii_mask


class HSLIC(AbstractMethod):
    def __init__(self, path: Path, ref_t: int, n_segments: int,
                 n_segments_h: int,
                 compactness: float, p_seeds_final: float = 0.01):
        super().__init__()
        self.n_segments_h = n_segments_h
        self.p_vol = None
        self.mri_spacing = None
        self.path_supervoxels = path
        self.ref_t = ref_t
        self.n_segments = n_segments
        self.compactness = compactness
        self.thread_safe = True
        self.p_seeds_final = p_seeds_final
        self.seeds = []

    def apply(self, patient: Patient, **kwargs):

        if True:
            mask = patient.get_image(self.ref_t).masks["pleural_region"].data
            n_seeds_final = self.calculate_seeds(patient, mask)

            # Get Spacing
            image = patient.get_image(self.ref_t).data
            nifti_args = patient.get_image(self.ref_t).nifti_props
            image_header = nifti_args["header"]
            self.mri_spacing = image_header.get_zooms()

            self.execute_h_slic(patient, image, mask, n_seeds_final)

            supervoxels_mask = self.combine_supervoxel_masks(patient, mask)

            save_nii_mask(patient=patient,
                          path=self.path_supervoxels,
                          mask=supervoxels_mask)

            # Update patient path masks
            patient.path_masks['supervoxels'] = self.path_supervoxels
        # except Exception as e:
        #     print("Error in id: ", patient.id)
        #     print(e)
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

    def execute_h_slic(self, patient, image, mask, n_seeds_final):
        stack = [mask]
        iteration_count = 0  # Safeguard against infinite loops
        max_iterations = 5000  # Set according to expected segmentation depth

        while stack:
            if iteration_count > max_iterations:
                print(
                    "Max iterations reached, breaking loop to avoid infinite recursion")
                break

            elif iteration_count == 0:
                msk = stack.pop()
                supervoxel_mask = slic(image=image, mask=mask,
                                       compactness=self.compactness,
                                       n_segments=n_seeds_final,
                                       channel_axis=None,
                                       start_label=1,
                                       spacing=self.mri_spacing)
                rps = skimage.measure.regionprops(supervoxel_mask)
                for rp in rps:
                    new_mask = np.zeros_like(msk)
                    slice_bbox = tuple(
                        slice(dim_start, dim_finish) for dim_start, dim_finish
                        in zip(rp.bbox[:3], rp.bbox[3:]))
                    lbl_in_bbox = rp.image
                    new_mask[slice_bbox] = lbl_in_bbox
                    if np.any(new_mask != msk):  # Ensure new mask is different
                        stack.append((new_mask))

            else:
                msk = stack.pop()
                supervoxel_mask = slic(image=image, mask=msk,
                                       compactness=self.compactness,
                                       n_segments=self.n_segments_h,
                                       channel_axis=None,
                                       start_label=1,
                                       spacing=self.mri_spacing)
                rps = skimage.measure.regionprops(supervoxel_mask)
                for rp in rps:
                    new_mask = np.zeros_like(msk)
                    slice_bbox = tuple(
                        slice(dim_start, dim_finish) for dim_start, dim_finish
                        in zip(rp.bbox[:3], rp.bbox[3:]))
                    lbl_in_bbox = rp.image
                    new_mask[slice_bbox] = lbl_in_bbox
                    if np.any(new_mask != msk):  # Ensure new mask is different
                        patient.supervoxels_m_masks.append(msk)
                        # stack.append((img, new_mask))
                    # print(f"New segment pushed with volume: {
                    # define_masks_volume(div_mask)}")

            iteration_count += 1

    @staticmethod
    def combine_supervoxel_masks(patient: Patient, pleural_mask):
        # combine the masks in supervoxel_mask,
        # so that the voxels in the same supervoxel have the same label
        supervoxel_mask = np.zeros_like(pleural_mask)
        if len(patient.supervoxels_m_masks) > 0:
            for i, s in enumerate(patient.supervoxels_m_masks):
                supervoxel_mask[s != 0] = i + 1
        return supervoxel_mask
