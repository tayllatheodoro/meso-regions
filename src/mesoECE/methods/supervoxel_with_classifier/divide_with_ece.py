from pathlib import Path
import numpy as np
import skimage
from skimage.segmentation import slic

from src.mesoECE.data_structure.patient import Patient
from src.mesoECE.methods import AbstractMethod
from src.mesoECE.methods.classifier.utils import superspels_labels_with_ece
from src.mesoECE.methods.superspels.utils import \
    define_mean_std_intensity_curves, save_and_plot_curves
from src.mesoECE.methods.utils import (define_masks_volume,
                                       setup_directories, save_nii_mask)


class DivideWithECE(AbstractMethod):
    def __init__(self, path: Path, ref_t: int, n_segments: int,
                 compactness: float, p_size: int = 2000,
                 method: str = None,
                 predict_only_small: bool = False):
        super().__init__()
        self.s_b_vol = None
        self.mri_spacing = None
        self.s_m_vol = None
        self.p_vol = None
        self.path_sv = path
        self.ref_t = ref_t
        self.n_segments = n_segments
        self.compactness = compactness
        self.thread_safe = True
        self.predicted_diagnosis = []
        self.method = method
        self.p_size = p_size
        self.predict_only_small = predict_only_small

    def apply(self, patient: Patient, **kwargs):
        if True:
            print(f"Processing patient {patient.id}...")
            setup_directories(path=self.path_sv,
                              dir_names=['ece_images', 'benign_images',
                                         'plots', 'curves_df'])
            # segmentation
            sv_m_mask, sv_b_mask = self.execute_div_ece(patient)

            # classification
            if self.significant_malignant_volume():
                self.process_malignant_case(patient, sv_m_mask)

            elif (not self.significant_malignant_volume() and
                  self.significant_benign_volume()):
                self.process_benign_case(patient)

            # # save benign supervoxels

            self.save_benign_supervoxels(patient, sv_b_mask)

        # except Exception as e:
        #     print(f"Error processing patient {patient.id}: {e}")

        new_patient = Patient(path=patient.path,
                              path_masks=patient.path_masks,
                              id=patient.id,
                              diagnosis=patient.diagnosis,
                              subclass_diagnosis=patient.subclass_diagnosis,
                              nodular=patient.nodular)
        return new_patient

    def result(self):
        return self.predicted_diagnosis

    def significant_benign_volume(self):
        return self.s_b_vol > 0

    def significant_malignant_volume(self):
        return self.s_m_vol > self.p_vol * 0.0001

    def save_benign_supervoxels(self, patient, mask):
        benign_curves, std_ece_curves = define_mean_std_intensity_curves(
            patient=patient,
            mask=mask)

        save_and_plot_curves(path=self.path_sv,
                             patient=patient,
                             curves=benign_curves,
                             curve_name='benign',
                             mask=mask)

        save_and_plot_curves(path=self.path_sv,
                             patient=patient,
                             curves=std_ece_curves,
                             curve_name='benign_std',
                             mask=mask)

        # Save supervoxels mask
        save_nii_mask(patient=patient,
                      path=self.path_sv / 'benign_images',
                      mask=mask)

    def process_benign_case(self, patient):

        if self.s_m_vol == 0:
            self.predicted_diagnosis.append(
                [patient.id, patient.diagnosis, 0,
                 patient.subclass_diagnosis, patient.nodular,
                 len(patient.supervoxels_m_masks), 0])
        else:
            self.predicted_diagnosis.append(
                [patient.id, patient.diagnosis, 0,
                 patient.subclass_diagnosis, patient.nodular,
                 len(patient.supervoxels_m_masks), self.s_m_vol])

    def process_malignant_case(self, patient, mask):
        self.predicted_diagnosis.append(
            [patient.id, patient.diagnosis, 1,
             patient.subclass_diagnosis, patient.nodular,
             len(patient.supervoxels_m_masks), self.s_m_vol])
        ece_curves, std_ece_curves = define_mean_std_intensity_curves(
            patient=patient,
            mask=mask)

        save_and_plot_curves(path=self.path_sv,
                             patient=patient,
                             curves=ece_curves,
                             curve_name='ece',
                             mask=mask)

        save_and_plot_curves(path=self.path_sv,
                             patient=patient,
                             curves=std_ece_curves,
                             curve_name='ece_std',
                             mask=mask)
        save_nii_mask(patient=patient,
                      path=self.path_sv / 'ece_images',
                      mask=mask)

    def execute_div_ece(self, patient: Patient):
        pleural_mask = patient.get_image(self.ref_t).masks[
            "pleural_region"].data.astype(np.int32)

        self.p_vol = define_masks_volume(pleural_mask)

        image = patient.get_image(self.ref_t).data
        nifti_args = patient.get_image(self.ref_t).nifti_props
        self.mri_spacing = nifti_args["header"].get_zooms()

        # Process the segmentation based on the `predict_only_small` flag
        if self.predict_only_small:
            self.segment_and_classify(patient, image, pleural_mask,
                                      self.n_segments,
                                      classify_immediately=False)
        else:
            self.segment_and_classify(patient, image, pleural_mask,
                                      self.n_segments,
                                      classify_immediately=True)

        sv_m_mask, sv_b_mask = self.combine_supervoxel_masks(
            patient,
            pleural_mask)
        self.s_m_vol = define_masks_volume(sv_m_mask)
        self.s_b_vol = define_masks_volume(sv_b_mask)

        return sv_m_mask, sv_b_mask

    def segment_and_classify(self, patient, image, mask, n_segments,
                             classify_immediately=False):
        stack = [(image, mask)]
        iteration_count = 0  # Safeguard against infinite loops
        max_iterations = 1000  # Set according to expected segmentation depth

        while stack:
            if iteration_count > max_iterations:
                print(
                    "Max iterations reached, breaking loop to avoid infinite recursion")
                break

            img, msk = stack.pop()
            current_volume = define_masks_volume(msk)
            # print(f"Processing segment with volume: {current_volume}")

            if current_volume <= self.p_size:
                diagnosis = self.predict_ece(patient, msk)
                # print(f"Segment classified with diagnosis: {diagnosis}")
                if diagnosis == 1:
                    patient.supervoxels_m_masks.append(msk)
                else:
                    patient.supervoxels_b_masks.append(msk)
            elif (len(stack) > 2 and
                  self.predict_ece(patient, msk) and classify_immediately):
                patient.supervoxels_m_masks.append(msk)
            else:
                supervoxel_mask = slic(image=img, mask=msk,
                                       compactness=self.compactness,
                                       n_segments=n_segments, channel_axis=None,
                                       start_label=1,
                                       spacing=self.mri_spacing)
                rps = skimage.measure.regionprops(supervoxel_mask)
                for rp in rps:
                    div_mask = np.zeros_like(msk)
                    slice_bbox = tuple(
                        slice(dim_start, dim_finish) for dim_start, dim_finish
                        in zip(rp.bbox[:3], rp.bbox[3:]))
                    lbl_in_bbox = rp.image
                    div_mask[slice_bbox] = lbl_in_bbox
                    if np.any(div_mask != msk) and define_masks_volume(
                            msk) > 1:  # Ensure new mask is different
                        stack.append((img, div_mask))
                    # print(f"New segment pushed with volume: {
                    # define_masks_volume(div_mask)}")

            iteration_count += 1

    def predict_ece(self, patient, mask):
        curves, std_curves = define_mean_std_intensity_curves(patient=patient,
                                                              mask=mask)
        ece_labels = superspels_labels_with_ece(
            index_270=patient.time_points.index(270),
            mean_intensity_curve=curves)

        s_vol = define_masks_volume(mask)

        return 1 if len(ece_labels) > 0 and s_vol > self.p_vol * 0.0001 else 0

    @staticmethod
    def combine_supervoxel_masks(patient: Patient, pleural_mask):
        # combine the masks in supervoxel_mask,
        # so that the voxels in the same supervoxel have the same label
        supervoxel_m_mask = np.zeros_like(pleural_mask)
        supervoxel_b_mask = np.zeros_like(pleural_mask)
        if len(patient.supervoxels_m_masks) > 0:
            for i, s in enumerate(patient.supervoxels_m_masks):
                supervoxel_m_mask[s != 0] = i + 1
        if len(patient.supervoxels_b_masks) > 0:
            for i, s in enumerate(patient.supervoxels_b_masks):
                supervoxel_b_mask[s != 0] = i + 1

        return supervoxel_m_mask, supervoxel_b_mask
