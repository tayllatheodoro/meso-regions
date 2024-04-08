import os
from pathlib import Path
import ants
import nibabel as nib
from tqdm import tqdm

from src.mesoECE.data_structure.mrimage import MRImage
from src.mesoECE.data_structure.patient import Patient
from src.mesoECE.methods import AbstractMethod


class Resample(AbstractMethod):
    def __init__(self, path: Path, ref_t: int = 270,
                 img_spacing: float = 1.78, img_dim=None):
        super().__init__()
        if img_dim is None:
            img_dim = [0, 0, 0]
        self.path_resample = path / 'images'
        self.path_resample_mask = path / 'masks'
        self.thread_safe = False
        self.img_spacing = img_spacing
        self.img_dim = img_dim
        self.ref_t = ref_t

    def apply(self, patient, **kwargs):

        # Creating directories to save resampled images and masks
        os.makedirs(self.path_resample, exist_ok=True)
        os.makedirs(self.path_resample_mask, exist_ok=True)
        # Getting images filenames
        filenames_temp = [MRImage.resolve_name(patient.id, t) for t in
                          patient.time_points]

        #  Checking if images are already corrected
        if os.listdir(self.path_resample):
            filenames_corrected = os.listdir(self.path_resample)
            filenames = [x for x in filenames_temp if
                         x not in filenames_corrected]
        else:
            filenames = filenames_temp

        # Applying resampling to all images from patient
        if len(filenames):
            for img_file in tqdm(filenames,
                                 desc="Resampling images from Patient: "
                                      + str(patient.id)):
                # Getting time point from image filename
                _, t = MRImage.parse_name(img_file)

                # Reading image as ANTsImage from numpy
                nifti_args = patient.get_image(self.ref_t).nifti_props
                image_ants = ants.from_nibabel(
                    nib.Nifti1Image(patient.get_image(t).data, **nifti_args))
                print("\n")
                print("Image spacing:", ants.get_spacing(image_ants))

                if self.img_spacing == 0:
                    # Resample the image to the image dim size
                    image_rsmp = ants.resample_image(image_ants,
                                                     (self.img_dim[0],
                                                      self.img_dim[1],
                                                      self.img_dim[2]),
                                                     True,
                                                     1)
                    print("Resampled image spacing:",
                          ants.get_spacing(image_rsmp))

                else:
                    # Resample the image to the target voxel size
                    image_rsmp = ants.resample_image(image_ants,
                                                     (self.img_spacing,
                                                      self.img_spacing,
                                                      self.img_spacing),
                                                     False,
                                                     1)
                    print("Resampled image spacing:",
                          ants.get_spacing(image_rsmp))

                # Checking if masks are available for the patient in t
                try:
                    # Resampling masks
                    for mask_type in patient.get_image(
                            self.ref_t).masks.keys():
                        try:
                            # Checking if masks type are available
                            print(
                                f'Obtaining Resampled MRI Mask {mask_type}...')
                            nifti_args_mask = patient.get_image(t).masks[
                                mask_type].nifti_props
                            os.makedirs(self.path_resample_mask / mask_type,
                                        exist_ok=True)

                            # Reading mask as ANTsImage from numpy
                            mask = ants.from_nibabel(
                                nib.Nifti1Image(
                                    patient.get_image(t).masks[mask_type].data,
                                    **nifti_args_mask))
                            print("Mask spacing:", ants.get_spacing(mask))

                            # Resample the mask to the image resampled size
                            mask_rsmp = ants.resample_image_to_target(mask,
                                                                      image_rsmp,
                                                                      interp_type='genericLabel')
                            print("Resampled mask spacing:",
                                  ants.get_spacing(mask_rsmp))
                            # Saving resampled mask
                            nib.save(ants.utils.convert_nibabel.to_nibabel(
                                mask_rsmp),
                                str(self.path_resample_mask / mask_type
                                    / img_file))
                        except:
                            print(
                                f'No mask {mask_type} found for patient'
                                f' {patient.id} at timepoint {t}')
                except:
                    print(
                        f'No masks found for patient {patient.id} at timepoint {t}')

                # Saving resampled image
                nib.save(ants.to_nibabel(image_rsmp),
                         str(self.path_resample / img_file))

        new_patient = Patient(path=self.path_resample,
                              path_masks=patient.path_masks,
                              id=patient.id,
                              diagnosis=patient.diagnosis,
                              subclass_diagnosis=patient.subclass_diagnosis,
                              nodular=patient.nodular)
        return new_patient


if __name__ == "__main__":
    import ants
