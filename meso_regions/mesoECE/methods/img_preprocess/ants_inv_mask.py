import os
from pathlib import Path
import ants

from meso_regions.mesoECE.data_structure.mrimage import MRImage
from meso_regions.mesoECE.data_structure.patient import Patient
from meso_regions.mesoECE.methods import AbstractMethod


class ANTsInvMask(AbstractMethod):
    def __init__(self, path: Path, ref_t: int = 270,
                 path_transforms: Path = None, **config):
        super().__init__()
        self.path_tfms = path_transforms
        self.ref_t = ref_t
        self.path_reg = path
        self.config = config
        self.thread_safe = False

    def apply(self, patient, **kwargs):
        mov_filenames = [MRImage.resolve_name(patient.id, t)
                                   for t in patient.time_points
                                   if t != self.ref_t]

        for mov_file in mov_filenames:

            # Reading moving image
            _, t = MRImage.parse_name(mov_file)

            # Saving transforms (forward and inverted) from the registrations
            img_ants = ants.image_read(str(patient.path / mov_file))
            tfm_inv_filename = f"{patient.id:05d}_{self.ref_t}_{t}.mat"

            # Applying inverse transform to obtain MRI images for each type
            # of mask
            for mask in patient.get_image(self.ref_t).masks.keys():
                if mov_file not in os.listdir(patient.path_masks[mask]):
                    print(
                        f'Applying inv transform to obtain MRI Mask {mask}...')

                    os.makedirs(patient.path_masks[mask], exist_ok=True)
                    mask_temp = ants.image_read(str(patient.path_masks[mask] /
                                                    patient.get_image(
                                                        self.ref_t).filename))
                    inv_mask = ants.apply_transforms(fixed=img_ants,
                                                     moving=mask_temp,
                                                     interpolator="genericLabel",
                                                     transformlist=[
                                                         str(self.path_tfms /
                                                             tfm_inv_filename)])
                    ants.image_write(inv_mask, str(
                        patient.path_masks[mask] / mov_file))

            new_patient = Patient(path=patient.path,
                                  path_masks=patient.path_masks,
                                  id=patient.id,
                                  diagnosis=patient.diagnosis,
                                  subclass_diagnosis=patient.subclass_diagnosis,
                                  nodular=patient.nodular)

        return new_patient


if __name__ == "__main__":
    import ants
