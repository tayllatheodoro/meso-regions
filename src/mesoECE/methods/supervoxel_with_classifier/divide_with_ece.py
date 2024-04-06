import nibabel as nib
import numpy as np
import skimage
from skimage.segmentation import slic
from pathlib import Path

from src.mesoECE.data_structure.patient import Patient
from src.mesoECE.methods import AbstractMethod

from src.mesoECE.methods.utils import define_masks_volume, \
    correct_image_background, correct_images_background
from src.mesoECE.methods.classifier.ece import superspels_labels_with_ece


class DivideWithECE(AbstractMethod):
    def __init__(self, path: Path, ref_t: int, n_segments: int,
                 compactness: float, p_size: int,
                 method: str = None,
                 domain: str = None,
                 predict_only_small: bool = False,
                 decrease_n_segments: bool = False):
        super().__init__()
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

    #TODO: Implement with sicle, disf
    #TODO: Implement plot and save supervoxels

    def apply(self, patient: Patient, **kwargs):
        try:
            path_m_images = self.path_sv / 'ece_images'
            path_b_images = self.path_sv / 'benign_images'
            path_plot = self.path_sv / 'plots'
            path_ss_df = self.path_sv / 'superspels_df'

            nifti_args = patient.get_image(self.ref_t).masks[
                "pleural_region"].nifti_props
            pleural_mask = patient.get_image(self.ref_t).masks[
                "pleural_region"].data.astype(np.int32)

            image = patient.get_image(self.ref_t).data

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

            else:
                self.predicted_diagnosis.append(
                    [patient.id, patient.diagnosis, 0,
                     patient.subclass_diagnosis, patient.nodular,
                     n_supervoxel, self.s_vol])

                nib.save(nib.Nifti1Image(sv_m_mask.astype(np.int32),
                                         **nifti_args),
                         str(path_m_images / patient.get_image(
                             self.ref_t).filename))
            # save benign supervoxels
            nib.save(nib.Nifti1Image(sv_b_mask.astype(np.int32),
                                     **nifti_args),
                     str(path_b_images / patient.get_image(
                         self.ref_t).filename))

            patient.path_masks['supervoxels'] = self.path_sv

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

            supervoxel_mask = slic(image, mask=mask,
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

            supervoxel_mask = slic(image, mask=mask,
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
        image_corrected = correct_images_background(patient)
        ss_mean = self.define_superspels_curves(patient,
                                                supervoxel_mask,
                                                image_corrected)
        ece_labels = superspels_labels_with_ece(
            index_270=patient.time_points.index(270),
            ss_mean=ss_mean)
        s_vol = define_masks_volume(mask=supervoxel_mask)
        if ece_labels and s_vol > self.p_vol * 0.0001:
            predicted_diagnosis = 1

        else:
            predicted_diagnosis = 0

        return predicted_diagnosis

    @staticmethod
    def define_superspels_curves(patient: Patient,
                                 supervoxel_mask,
                                 image_corrected):
        rps = skimage.measure.regionprops(supervoxel_mask)
        ss_mean_curves = np.zeros((len(rps), len(patient.time_points)))

        for rp in rps:
            slice_bbox = tuple(
                [slice(dim_start, dim_finish) for dim_start, dim_finish in
                 zip(rp.bbox[:3], rp.bbox[3:])])
            lbl_in_bbox = rp.image

            for t in patient.time_points:
                img = image_corrected[patient.time_points.index(t)]
                img_in_bbox = img[slice_bbox]
                ss_mean_curves[rp.label - 1, patient.time_points.index(t)] = \
                    img_in_bbox[lbl_in_bbox > 0].mean()
        return ss_mean_curves

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
