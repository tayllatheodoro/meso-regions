import os
from pathlib import Path
import ants

from src.mesoECE.data_structure.mrimage import MRImage
from src.mesoECE.data_structure.patient import Patient
from src.mesoECE.methods import AbstractMethod


class ANTsInvMask(AbstractMethod):
    def __init__(self, path: Path, ref_t: int = 270, path_transforms: Path = None, **config):
        super().__init__()
        self.path_transforms = Path(path_transforms)
        self.reference_t = ref_t
        self.path_registration = path
        self.config = config
        self.thread_safe = False

    def apply(self, patient, **kwargs):
        moving_images_filenames = [MRImage.resolve_name(patient.id, t) for t in patient.time_points if t != self.reference_t]

        for mov_img_filename in moving_images_filenames:

            # Reading moving image
            _, t = MRImage.parse_name(mov_img_filename)

            # Saving transforms (forward and inverted) from the registrations
            orig_img_ants = ants.image_read(str(patient.path / mov_img_filename))
            transform_inv_filename = f"{patient.id:05d}_{self.reference_t}_{t}.mat"

            # Applying inverse transform to obtain MRI images for each type
            # of mask
            for mask in patient.get_image(self.reference_t).masks.keys():
                if mov_img_filename not in os.listdir(patient.path_masks[mask]):

                    print(f'Applying inv transform to obtain MRI Mask {mask}...')

                    os.makedirs(patient.path_masks[mask], exist_ok=True)
                    mask_temp = ants.image_read(str(patient.path_masks[mask] / patient.get_image(self.reference_t).filename))
                    inv_mask = ants.apply_transforms(fixed=orig_img_ants,
                                                 moving=mask_temp,
                                                 interpolator="genericLabel",
                                                 transformlist=[str(self.path_transforms / transform_inv_filename)])
                    ants.image_write(inv_mask, str(patient.path_masks[mask] / mov_img_filename))


            new_patient = Patient(path=patient.path,
                                  path_masks=patient.path_masks,
                                  id=patient.id, diagnosis=patient.diagnosis,
                                  subclass_diagnosis=patient.subclass_diagnosis,
                                  nodular=patient.nodular)

        return new_patient



if __name__ == "__main__":
    import ants