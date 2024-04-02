import nibabel as nib
from pathlib import Path
import numpy as np
from skimage.filters import threshold_otsu
from skimage.morphology import binary_dilation, ball

from src.mesoECE.data_structure.patient import Patient
from src.mesoECE.methods import AbstractMethod


def otsu_threshold_mask(img, mask):
    # Appy otsu in the masked image
    img_masked = img.copy()
    img_masked[mask != 1] = 0
    otsu_threshold = threshold_otsu(img_masked)
    mask_binary = img_masked > otsu_threshold
    mask_otsu = mask_binary.astype(np.int32)
    # return a label mask with img_otsu
    return mask_otsu


class Dilation(AbstractMethod):
    def __init__(self, path: Path, ref_t: int, dilation_radius: int,
                 p_center_distance: float = 0.5, otsu: bool = False, mask_to_dilate: str = 'fluid'):
        super().__init__()
        self.path_pleural_region = path
        self.ref_t = ref_t
        self.thread_safe = True
        self.dilation_radius = dilation_radius
        self.otsu = otsu
        self.p_center_distance = p_center_distance
        self.mask_to_dilate = mask_to_dilate

    def apply(self, patient: Patient, **kwargs):
        try:
            nifti_args = patient.get_image(self.ref_t).get_mask(
                self.mask_to_dilate).nifti_props
            mask = patient.get_image(self.ref_t).get_mask(
                self.mask_to_dilate).data
            img = patient.get_image(self.ref_t).data

            if self.p_center_distance > 0:
                mask_dilated = self.dilate_mask_far_from_center(mask)
            else:
                mask_dilated = self.dilate_mask(mask)

            # Subtraction of the dilated mask with the original mask
            mask_sub = np.logical_xor(mask_dilated, mask).astype(np.int32)

            if self.otsu:
                mask_sub = otsu_threshold_mask(img, mask_sub)
            patient.path_masks['pleural_region'] = self.path_pleural_region
            nib.save(nib.Nifti1Image(mask_sub, **nifti_args),
                     str(self.path_pleural_region / patient.get_image(
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

    def dilate_mask(self, mask):
        return binary_dilation(mask, ball(self.dilation_radius))

    def dilate_mask_far_from_center(self, mask):
        # Find the center of the image
        center = np.array(mask.shape) / 2

        # Create a distance map
        coords = np.ogrid[[slice(0, dim) for dim in mask.shape]]
        distance_map = np.sqrt(
            sum([(coord - cen) ** 2 for coord, cen in zip(coords, center)]))

        # Find the maximum distance within the mask from the center
        max_distance_in_mask = np.max(distance_map[mask > 0])
        # min_distance_in_mask = np.min(distance_map[mask > 0])
        # Set the threshold to the difference between the max and min distances
        # threshold_distance = max_distance_in_mask - (min_distance_in_mask)

        # Set the threshold to half of this maximum distance
        threshold_distance = max_distance_in_mask * self.p_center_distance

        # Apply conditional dilation
        mask_temp = distance_map > threshold_distance
        mask_dilated = np.where(mask_temp, self.dilate_mask(mask), mask)

        return mask_dilated
