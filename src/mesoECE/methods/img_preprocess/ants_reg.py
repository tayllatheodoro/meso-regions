import os
from pathlib import Path
import ants
import nibabel as nib
from tqdm import tqdm

from src.mesoECE.data_structure.mrimage import MRImage
from src.mesoECE.data_structure.patient import Patient
from src.mesoECE.methods import AbstractMethod


class ANTs(AbstractMethod):
    def __init__(self, path: Path, ref_t: int = 270, domain: str = None,
                 **config):
        super().__init__()
        self.domain = domain
        self.thread_safe = False
        self.ref_t = ref_t
        self.path_reg = path
        self.config = config
        self.path_reg_images = self.path_reg / "images"
        self.path_reg_tfms = self.path_reg / "ants_transforms"

    def apply(self, patient, **kwargs):
        # defining the fix_image_filename image and the moving targets
        # images for registration
        fix_filename = MRImage.resolve_name(patient.id, self.ref_t)
        mov_filenames_temp = [MRImage.resolve_name(patient.id, t)
                              for t in patient.time_points if
                              t != self.ref_t]

        if os.path.exists(self.path_reg / "images"):
            mov_filenames_corrected = os.listdir(self.path_reg / "images")

            mov_filenames = [x for x in mov_filenames_temp
                             if x not in mov_filenames_corrected]
        else:
            mov_filenames = mov_filenames_temp

        if len(mov_filenames):
            # making dir to save transform matrix from the registrations
            os.makedirs(self.path_reg_images, exist_ok=True)
            os.makedirs(self.path_reg_tfms, exist_ok=True)

            # reading image as ANTsImage from numpy and saving reference image
            nifti_args = patient.get_image(self.ref_t).nifti_props
            fix_img_ants = ants.from_nibabel(
                nib.Nifti1Image(patient.get_image(self.ref_t).data,
                                **nifti_args))
            nib.save(ants.to_nibabel(fix_img_ants),
                     str(self.path_reg_images / fix_filename))

            print("\n")
            for mov_file in tqdm(mov_filenames,
                                 desc="Registering images from "
                                      "Patient: " + str(patient.id)):
                # reading moving image

                id, t = MRImage.parse_name(mov_file)
                nifti_args_mov = patient.get_image(t).nifti_props

                mov_img_ants = ants.from_nibabel(
                    nib.Nifti1Image(patient.get_image(t).data,
                                    **nifti_args_mov))

                # performing registration

                print(f"{mov_file} to {fix_filename}...")
                reg = ants.registration(fixed=fix_img_ants,
                                        moving=mov_img_ants,
                                        **self.config)

                # saving registered outputs
                warped_fix_img = ants.to_nibabel(reg["warpedfixout"])
                warped_mov_img = ants.to_nibabel(reg["warpedmovout"])  ## ECE

                tfm_fwd_1 = ants.read_transform(reg["fwdtransforms"][1])

                vec_fwd = ants.image_read(reg['fwdtransforms'][0])
                atx_fwd = ants.transform_from_displacement_field(vec_fwd)

                tfm_fwd = ants.compose_ants_transforms([tfm_fwd_1, atx_fwd])

                tfm_inv_1 = ants.read_transform(reg["invtransforms"][0])

                vec_inv = ants.image_read(reg['invtransforms'][1])
                atx_inv = ants.transform_from_displacement_field(vec_inv)

                tfm_inv = ants.compose_ants_transforms([tfm_inv_1, atx_inv])

                nib.save(warped_mov_img,
                         str(self.path_reg_images / mov_file))

                # saving transforms (forward and inverted) from the registration
                tfm_fwd_filename = f"{patient.id:05d}_{t}_{self.ref_t}.mat"
                tfm_inv_filename = f"{patient.id:05d}_{self.ref_t}_{t}.mat"

                ants.write_transform(tfm_fwd,
                                     str(self.path_reg_tfms / tfm_fwd_filename))
                ants.write_transform(tfm_inv,
                                     str(self.path_reg_tfms / tfm_inv_filename))

        new_patient = Patient(path=self.path_reg_images,
                              path_masks=patient.path_masks,
                              id=patient.id,
                              diagnosis=patient.diagnosis,
                              subclass_diagnosis=patient.subclass_diagnosis,
                              nodular=patient.nodular)

        return new_patient


if __name__ == "__main__":
    import ants
