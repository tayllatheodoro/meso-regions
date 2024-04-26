import os

import numpy as np
import skimage
from skimage.segmentation import slic
from pathlib import Path
from src.mesoECE.data_structure.patient import Patient
from src.mesoECE.methods import AbstractMethod
from src.mesoECE.methods.superspels.utils import define_mean_intensity_curves, \
    save_curves_and_interp_to_csv, save_superspels_masks, plot_curves

from src.mesoECE.methods.utils import (define_masks_volume,
                                       correct_images_background, setup_directories)
from src.mesoECE.methods.classifier.utils import superspels_labels_with_ece
import nibabel as nib


class DivideWithECE(AbstractMethod):
    def __init__(self, path: Path, ref_t: int, n_segments: int,
                 compactness: float, p_size: int = 2000,
                 method: str = None,
                 domain: str = None,
                 predict_only_small: bool = False):
        super().__init__()
        self.mri_spacing = None
        self.images_corrected = None
        self.s_vol = None
        self.p_vol = None
        self.path_sv = path
        self.domain = domain
        self.ref_t = ref_t
        self.n_segments = n_segments
        self.compactness = compactness
        self.thread_safe = False
        self.predicted_diagnosis = []
        self.method = method
        self.p_size = p_size
        self.predict_only_small = predict_only_small

    def apply(self, patient: Patient, **kwargs):
        try:
            print(f"Processing patient {patient.id}...")
            setup_directories(path=self.path_sv,
                              dir_names=['ece_images', 'benign_images',
                                         'plots', 'curves_df'])
            pleural_mask = patient.get_image(self.ref_t).masks["pleural_region"].data.astype(np.int32)
            image = patient.get_image(self.ref_t).data
            nifti_args = patient.get_image(self.ref_t).nifti_props
            self.mri_spacing = nifti_args["header"].get_zooms()

            # Process the segmentation based on the `predict_only_small` flag
            if self.predict_only_small:
                self.segment_and_classify(patient, image, pleural_mask, self.n_segments, classify_immediately=False)
            else:
                self.segment_and_classify(patient, image, pleural_mask, self.n_segments, classify_immediately=True)

            sv_m_mask, sv_b_mask = self.combine_supervoxel_masks(
                patient,
                pleural_mask)

            # classification

            n_supervoxel = len(patient.supervoxels_m_masks)
            if n_supervoxel > 0:
                self.s_vol = define_masks_volume(sv_m_mask)

                self.predicted_diagnosis.append(
                    [patient.id, patient.diagnosis, 1,
                     patient.subclass_diagnosis, patient.nodular,
                     n_supervoxel, self.s_vol])

                ece_curves, std_ece_curves = define_mean_intensity_curves(
                    patient=patient,
                    mask=sv_m_mask,
                    domain=self.domain)

                save_curves_and_interp_to_csv(patient=patient,
                                              curves=std_ece_curves,
                                              path=self.path_sv / 'curves_df',
                                              curve_name='ece')
                save_curves_and_interp_to_csv(patient=patient,
                                              curves=ece_curves,
                                              path=self.path_sv / 'curves_df',
                                              curve_name='ece')

                # # Plot mean intensity curves
                plot_curves(curve=ece_curves,
                            time_points=patient.time_points,
                            mask=sv_m_mask,
                            filename=str(self.path_sv / 'plots' / f'ece_{patient.id}.png'),
                            mean_plot=True)

                # Save supervoxels mask
                nib.save(nib.Nifti1Image(sv_m_mask.astype(np.int32),
                                         **nifti_args),
                         str(self.path_sv / 'ece_images' / patient.get_image(
                             self.ref_t).filename))


            else:
                self.predicted_diagnosis.append(
                    [patient.id, patient.diagnosis, 0,
                     patient.subclass_diagnosis, patient.nodular,
                     n_supervoxel, self.s_vol])

            # # save benign supervoxels
            #
            benign_curves, std_ece_curves = define_mean_intensity_curves(
                patient=patient,
                mask=sv_b_mask,
                domain=self.domain)

            # patient.curves['benign'] = benign_curves
            save_curves_and_interp_to_csv(patient=patient,
                                          curves=benign_curves,
                                          path=self.path_sv / 'curves_df',
                                          curve_name='benign')
            save_curves_and_interp_to_csv(patient=patient,
                                          curves=std_ece_curves,
                                          path=self.path_sv / 'curves_df',
                                          curve_name='std_benign')
            # Save supervoxels mask
            nib.save(nib.Nifti1Image(sv_b_mask.astype(np.int32),
                                     **nifti_args),
                     str(self.path_sv / 'benign_images' / patient.get_image(
                         self.ref_t).filename))

            plot_curves(curve=benign_curves,
                        time_points=patient.time_points,
                        mask=sv_b_mask,
                        filename=str(self.path_sv / 'plots' / f'benign_{patient.id}.png'),
                        mean_plot=True,
                        title='Benign Curves')


        except Exception as e:
            print(f"Error processing patient {patient.id}: {e}")

        new_patient = Patient(path=patient.path,
                              path_masks=patient.path_masks,
                              id=patient.id,
                              diagnosis=patient.diagnosis,
                              subclass_diagnosis=patient.subclass_diagnosis,
                              nodular=patient.nodular)
        return new_patient

    def result(self):
        return self.predicted_diagnosis

    def segment_and_classify(self, patient, image, mask, n_segments, classify_immediately):
        stack = [(image, mask)]
        while stack:
            img, msk = stack.pop()
            if define_masks_volume(msk) <= self.p_size or classify_immediately:
                # Direct classification and storage decision based on predict_ece outcome
                diagnosis = self.predict_ece(patient, msk)
                if diagnosis == 1:
                    patient.supervoxels_m_masks.append(msk)
                else:
                    patient.supervoxels_b_masks.append(msk)
            else:
                # Perform further segmentation
                supervoxel_mask = slic(image=img, mask=msk, compactness=self.compactness,
                                       n_segments=n_segments, channel_axis=None, start_label=1,
                                       spacing=self.mri_spacing)
                rps = skimage.measure.regionprops(supervoxel_mask)
                for rp in rps:
                    div_mask = np.zeros_like(msk)
                    slice_bbox = tuple(
                        slice(dim_start, dim_finish) for dim_start, dim_finish in zip(rp.bbox[:3], rp.bbox[3:]))
                    lbl_in_bbox = rp.image
                    div_mask[slice_bbox] = lbl_in_bbox
                    stack.append((img, div_mask))

    def predict_ece(self, patient, mask):
        curves = define_mean_intensity_curves(patient=patient, mask=mask, domain=self.domain)
        ece_labels = superspels_labels_with_ece(index_270=patient.time_points.index(270),
                                                mean_intensity_curve=curves[0])
        s_vol = define_masks_volume(mask)
        return 1 if len(ece_labels) > 0 and s_vol > self.p_size * 0.0001 else 0

    @staticmethod
    def combine_supervoxel_masks(patient: Patient, pleural_mask):
        # combine the masks in supervoxel_mask,
        # so that the voxels in the same supervoxel have the same label
        supervoxel_m_mask = np.zeros_like(pleural_mask)
        supervoxel_b_mask = np.zeros_like(pleural_mask)
        if len(patient.supervoxels_m_masks) > 0:
            for i, s in enumerate(patient.supervoxels_m_masks):
                supervoxel_m_mask[s != 0] = i + 1
        for i, s in enumerate(patient.supervoxels_b_masks):
            supervoxel_b_mask[s != 0] = i + 1

        return supervoxel_m_mask, supervoxel_b_mask
