import nibabel as nib
import numpy as np
import skimage
from skimage.segmentation import slic
from pathlib import Path
from src.mesoECE.data_structure.patient import Patient
from src.mesoECE.methods import AbstractMethod
from src.mesoECE.methods.superspels.utils import define_mean_intensity_curves, \
    save_curves_and_interp_to_csv, save_superspels_masks, plot_curves

from src.mesoECE.methods.utils import (define_masks_volume,
                                       correct_images_background)
from src.mesoECE.methods.classifier.utils import superspels_labels_with_ece


class DivideWithECE(AbstractMethod):
    def __init__(self, path: Path, ref_t: int, n_segments: int,
                 compactness: float, p_size: int,
                 method: str = None,
                 domain: str = None,
                 predict_only_small: bool = False,
                 decrease_n_segments: bool = False):
        super().__init__()
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
        self.decrease_n_segments = decrease_n_segments
        self.method = method
        self.p_size = p_size
        self.predict_only_small = predict_only_small

    # TODO: Implement with sicle, disf
    def apply(self, patient: Patient, **kwargs):
        try:
            path_m_images = self.path_sv / 'ece_images'
            path_b_images = self.path_sv / 'benign_images'
            path_plot = self.path_sv / 'plots'
            path_df = self.path_sv / 'curves_df'

            pleural_mask = patient.get_image(self.ref_t).masks[
                "pleural_region"].data.astype(np.int32)

            image = patient.get_image(self.ref_t).data
            self.images_corrected = correct_images_background(patient)

            if self.predict_only_small:
                self.divide_with_ece_small_slic(patient, image,
                                                pleural_mask,
                                                self.n_segments)
            else:
                self.divide_with_ece_slic(patient,
                                          image,
                                          pleural_mask,
                                          self.n_segments)

            sv_m_mask, sv_b_mask = self.combine_supervoxel_masks(
                patient,
                pleural_mask)
            self.p_vol = define_masks_volume(pleural_mask)
            self.s_vol = define_masks_volume(sv_m_mask)

            # classification

            n_supervoxel = len(patient.supervoxels_m_masks)
            if n_supervoxel == 0 or self.s_vol > self.p_vol * 0.0001:
                self.predicted_diagnosis.append(
                    [patient.id, patient.diagnosis, 1,
                     patient.subclass_diagnosis, patient.nodular,
                     n_supervoxel, self.s_vol])

                ece_curves = define_mean_intensity_curves(
                    patient=patient,
                    mask=sv_m_mask,
                    images_corrected=self.images_corrected,
                    domain=self.domain)

                patient.curves['ece'] = ece_curves

                save_curves_and_interp_to_csv(patient=patient,
                                              curves=ece_curves,
                                              path=path_df,
                                              curve_name='ece')

                # Plot mean intensity curves
                plot_curves(curve=ece_curves,
                            time_points=patient.time_points,
                            mask=sv_m_mask,
                            filename=str(path_plot / f'ece_{patient.id}.png'),
                            mean_plot=True)

                save_superspels_masks(mask=sv_m_mask,
                                      nifti_args=patient.nifti_args,
                                      patient=patient,
                                      domain=self.domain,
                                      ref_t=self.ref_t,
                                      path=path_m_images)

            else:
                self.predicted_diagnosis.append(
                    [patient.id, patient.diagnosis, 0,
                     patient.subclass_diagnosis, patient.nodular,
                     n_supervoxel, self.s_vol])

            # save benign supervoxels

            benign_curves = define_mean_intensity_curves(
                patient=patient,
                mask=sv_b_mask,
                images_corrected=self.images_corrected,
                domain=self.domain)

            patient.curves['benign'] = benign_curves
            save_curves_and_interp_to_csv(patient=patient,
                                          curves=benign_curves,
                                          path=path_df,
                                          curve_name='benign')
            save_superspels_masks(mask=sv_b_mask,
                                  nifti_args=patient.nifti_args,
                                  patient=patient,
                                  domain=self.domain,
                                  ref_t=self.ref_t,
                                  path=path_b_images)

            plot_curves(curve=benign_curves,
                        time_points=patient.time_points,
                        mask=sv_b_mask,
                        filename=str(path_plot / f'benign_{patient.id}.png'),
                        mean_plot=True,
                        title='Benign Curves')

            patient.path_masks['supervoxels_with_ece'] = self.path_sv

        except:
            print("Error in patient:", patient.id)
        new_patient = Patient(path=patient.path,
                              path_masks=patient.path_masks,
                              id=patient.id,
                              diagnosis=patient.diagnosis,
                              subclass_diagnosis=patient.subclass_diagnosis,
                              nodular=patient.nodular)
        return new_patient

    def results(self):
        return self.predicted_diagnosis

    def divide_with_ece_small_slic(self, patient: Patient, image, mask,
                                   n_segments):
        if np.sum(mask) >= self.p_vol * self.p_size:
            print("\r", end='')
            print("Dividing...", end="", flush=True)

            supervoxel_mask = slic(image=image,
                                   mask=mask,
                                   compactness=self.compactness,
                                   n_segments=n_segments)
            rps = skimage.measure.regionprops(supervoxel_mask)

            if self.decrease_n_segments:
                if n_segments <= 2:
                    n_segments = 2
                else:
                    n_segments = n_segments // 2

            for rp in rps:
                div_mask = np.zeros_like(mask)

                slice_bbox = tuple(
                    [slice(dim_start, dim_finish) for dim_start, dim_finish in
                     zip(rp.bbox[:3], rp.bbox[3:])])
                lbl_in_bbox = rp.image
                div_mask[slice_bbox] = lbl_in_bbox

                self.divide_with_ece_small_slic(patient, image, div_mask,
                                                n_segments)

        elif (self.predict_ece(patient, mask) == 1 and
              np.sum(mask) < self.p_vol * self.p_size):
            patient.supervoxels_m_masks.append(mask)
        elif (self.predict_ece(patient, mask) == 0 and
              np.sum(mask) < self.p_vol * self.p_size):
            patient.supervoxels_b_masks.append(mask)

    def divide_with_ece_slic(self, patient: Patient, image, mask,
                             n_segments):
        if (self.predict_ece(patient, mask) == 0 and
                np.sum(mask) >= self.p_vol * self.p_size):
            print("\r", end='')
            print("Dividing...", end="", flush=True)

            supervoxel_mask = slic(image=image,
                                   mask=mask,
                                   compactness=self.compactness,
                                   n_segments=n_segments)
            rps = skimage.measure.regionprops(supervoxel_mask)
            if self.decrease_n_segments:
                if n_segments <= 2:
                    n_segments = 2
                else:
                    n_segments = n_segments // 2

            for rp in rps:
                div_mask = np.zeros_like(mask)

                slice_bbox = tuple(
                    [slice(dim_start, dim_finish) for dim_start, dim_finish in
                     zip(rp.bbox[:3], rp.bbox[3:])])
                lbl_in_bbox = rp.image
                div_mask[slice_bbox] = lbl_in_bbox

                self.divide_with_ece_small_slic(patient, image, div_mask,
                                                n_segments)

        elif self.predict_ece(patient, mask) == 1:
            patient.supervoxels_m_masks.append(mask)
        elif (self.predict_ece(patient, mask) == 0 and
              np.sum(mask) < self.p_vol * self.p_size):
            patient.supervoxels_b_masks.append(mask)

    def predict_ece(self, patient: Patient, supervoxel_mask):

        mean_intensity_curve = define_mean_intensity_curves(
            patient=patient,
            mask=supervoxel_mask,
            images_corrected=self.images_corrected,
            domain=self.domain)

        ece_labels = superspels_labels_with_ece(
            index_270=patient.time_points.index(270),
            mean_intensity_curve=mean_intensity_curve)

        s_vol = define_masks_volume(mask=supervoxel_mask)

        if ece_labels and s_vol > self.p_vol * 0.0001:
            predicted_diagnosis = 1

        else:
            predicted_diagnosis = 0

        return predicted_diagnosis

    @staticmethod
    def combine_supervoxel_masks(patient: Patient, pleural_mask):
        # combine the masks in supervoxel_mask,
        # so that the voxels in the same supervoxel have the same label
        supervoxel_m_mask = np.zeros_like(pleural_mask)
        supervoxel_b_mask = np.zeros_like(pleural_mask)
        for i, s in enumerate(patient.supervoxels_m_masks):
            supervoxel_m_mask[s != 0] = i + 1
        for i, s in enumerate(patient.supervoxels_b_masks):
            supervoxel_b_mask[s != 0] = i + 1

        return supervoxel_m_mask, supervoxel_b_mask
