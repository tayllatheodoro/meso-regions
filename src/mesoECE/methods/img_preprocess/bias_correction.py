import os
from pathlib import Path
import ants
import nibabel as nib

from src.mesoECE.data_structure.mrimage import MRImage
from src.mesoECE.data_structure.patient import Patient
from src.mesoECE.methods import AbstractMethod


class N4BiasCorrection(AbstractMethod):
    def __init__(self, path: Path):
        super().__init__()
        self.path_bias_correction = path / 'images'
        self.thread_safe = False

    def apply(self, patient, **kwargs):

        # Getting images filenames
        os.makedirs(self.path_bias_correction, exist_ok=True)
        filenames_temp = [MRImage.resolve_name(patient.id, t) for t in
                          patient.time_points]

        # Checking if images are already corrected
        if os.path.exists(self.path_bias_correction):
            filenames_corrected = os.listdir(self.path_bias_correction)
            filenames = [x for x in filenames_temp if
                         x not in filenames_corrected]
        else:
            filenames = filenames_temp

        # Applying N4 Bias Correction to all images from patient

        if len(filenames):
            for img_file in filenames:
                # Getting time point from image filename
                _, t = MRImage.parse_name(img_file)

                # Reading image as ANTsImage from numpy nifti_props are the
                # parameters of header and affine from the original image
                # needed to keep voxel size and orientation
                nifti_args = patient.get_image(t).nifti_props
                image_ants = ants.from_nibabel(
                    nib.Nifti1Image(patient.get_image(t).data, **nifti_args))

                # Performing N4 Bias Correction
                image_corrected = ants.n4_bias_field_correction(image_ants)

                # Saving N4 Bias Correction image
                nib.save(ants.to_nibabel(image_corrected),
                         str(self.path_bias_correction / img_file))

        new_patient = Patient(path=self.path_bias_correction,
                              path_masks=patient.path_masks,
                              id=patient.id,
                              diagnosis=patient.diagnosis,
                              subclass_diagnosis=patient.subclass_diagnosis,
                              nodular=patient.nodular)
        return new_patient


if __name__ == "__main__":
    import ants
