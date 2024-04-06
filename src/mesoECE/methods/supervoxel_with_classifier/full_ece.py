import numpy as np
import os

import skimage.measure

from scipy.ndimage import uniform_filter
from pathlib import Path

from src.mesoECE.data_structure import MRImage
from src.mesoECE.data_structure.patient import Patient
from src.mesoECE.methods import AbstractMethod
from src.mesoECE.methods.classifier.utils import plot_curves, \
    save_superspels_masks, save_curves_and_interp_to_csv
from src.mesoECE.methods.utils import (define_masks_volume,
                                       correct_image_background,
                                       correct_images_background)


class FullECE(AbstractMethod):
    def __init__(self, path: Path, ref_t: int, filter_size: int = 5,
                 with_mask: bool = True, domain: str = 'REG'):
        super().__init__()
        self.domain = None
        self.thread_safe = True
        self.predicted_diagnosis = []
        self.path_ece = path
        self.ref_t = ref_t
        self.filter_size = filter_size
        self.with_mask = with_mask

    def apply(self, patient: Patient, **kwargs):
        try:
            path_m_images = self.path_ece / 'ece_images'
            path_b_images = self.path_ece / 'benign_images'
            path_plot = self.path_ece / 'plots'
            path_ss_df = self.path_ece / 'superspels_df'

            os.makedirs(path_m_images, exist_ok=True)
            os.makedirs(path_b_images, exist_ok=True)
            os.makedirs(path_ss_df, exist_ok=True)
            os.makedirs(path_plot, exist_ok=True)

            nifti_args = patient.get_image(self.ref_t).nifti_props

            # Correct images background
            images_corrected = correct_images_background(patient=patient)

            # Apply mean filter to simulate mean of superspels
            images = self.define_images_filtered(
                patient=patient,
                pleural_mask=pleural_mask,
                images_corrected=images_corrected)

            # Define mask with superspels with ece pattern
            # and benign masks
            ece_mask, benign_mask = self.define_full_ece_mask(patient, images)

            # label the masks
            ece_labeled_mask = skimage.measure.label(ece_mask)
            benign_labeled_mask = skimage.measure.label(benign_mask)

            # Calculate the volume of the pleural mask
            pleural_mask = patient.get_image(self.ref_t).masks[
                "pleural_region"].data.astype(np.int32)
            pleural_vol = define_masks_volume(mask=pleural_mask)

            # Calculate the volume of the ECE mask
            ece_vol = define_masks_volume(mask=ece_mask)

            # if there are superspels with ece pattern and the volume of the
            # ece mask is greater than 0.01% of the pleural mask volume to
            # reduce false positives

            if ece_vol > pleural_vol * 0.0001:
                self.predicted_diagnosis.append(
                    [patient.id, patient.diagnosis, 1,
                     patient.subclass_diagnosis, patient.nodular,
                     ece_vol, ece_vol])

                ece_curves = self.define_ece_curves(patient,
                                                    ece_labeled_mask)

                save_curves_and_interp_to_csv(patient=patient,
                                              curves=ece_curves,
                                              ref_t=self.ref_t,
                                              path=path_ss_df,
                                              curve_name='ece')
                plot_curves(curve=ece_curves,
                            mask=ece_labeled_mask,
                            time_points=patient.time_points,
                            filename=str(path_plot /
                                         MRImage.resolve_name(
                                             patient.id, self.ref_t,
                                             "png")))
                plot_curves(curve=ece_curves,
                            mask=ece_labeled_mask,
                            time_points=patient.time_points,
                            filename=str(path_plot /
                                         MRImage.resolve_name(
                                             patient.id, self.ref_t,
                                             "png")))

                save_superspels_masks(ss_mask=ece_labeled_mask,
                                      nifti_args=nifti_args,
                                      patient=patient,
                                      domain=self.domain,
                                      ref_t=self.ref_t,
                                      path=path_m_images)

                save_superspels_masks(ss_mask=benign_labeled_mask,
                                      nifti_args=nifti_args,
                                      patient=patient,
                                      domain=self.domain,
                                      ref_t=self.ref_t,
                                      path=path_b_images)

            else:
                self.predicted_diagnosis.append(
                    [patient.id, patient.diagnosis, 0,
                     patient.subclass_diagnosis, patient.nodular,
                     ece_vol, ece_vol])

            save_superspels_masks(ss_mask=benign_labeled_mask,
                                  nifti_args=nifti_args,
                                  patient=patient,
                                  domain=self.domain,
                                  ref_t=self.ref_t,
                                  path=path_b_images)

            patient.path_masks['ece'] = self.path_ece
        except:
            print("Error in id: ", patient.id)
        new_patient = Patient(path=patient.path,
                              path_masks=patient.path_masks,
                              id=patient.id,
                              diagnosis=patient.diagnosis,
                              subclass_diagnosis=patient.subclass_diagnosis,
                              nodular=patient.nodular)
        return new_patient

    def results(self):
        return self.predicted_diagnosis

    def define_images_filtered(self, patient, pleural_mask, images_corrected):
        images = []
        for t in patient.time_points:
            # image mean filter using skitmage
            img_filtered = uniform_filter(
                input=images_corrected[patient.time_points.index(t)],
                size=self.filter_size)
            if self.with_mask:
                img_filtered[pleural_mask == 0] = 0
            images.append(img_filtered)
        return images

    def define_full_ece_mask(self, patient, images):
        ece_mask = np.ones_like(patient.get_image(self.ref_t).data.shape)
        peak_index = patient.time_points.index(270)
        for i, t in enumerate(patient.time_points):
            if i < peak_index:
                ece_mask = np.logical_and(ece_mask, images[i] < images[i + 1])
            elif i > peak_index:
                ece_mask = np.logical_and(ece_mask, images[i] < images[i - 1])
        benign_mask = np.logical_not(ece_mask)
        return ece_mask, benign_mask

    @staticmethod
    def define_ece_curves(patient: Patient, ece_mask):

        ece_mean_curve = np.zeros(ece_mask.max() + 1,
                                  patient.time_points.__len__())

        rps = skimage.measure.regionprops(ece_mask)
        for rp in rps:
            slice_bbox = tuple(
                [slice(dim_start, dim_finish) for dim_start, dim_finish in
                 zip(rp.bbox[:3], rp.bbox[3:])])
            lbl_in_bbox = rp.image

            for t in patient.time_points:
                img = correct_image_background(patient, t)
                img_in_bbox = img[slice_bbox]
                ece_mean_curve[rp.label, patient.time_points.index(t)] = \
                    img_in_bbox[lbl_in_bbox > 0]
        benign_curves = ece_mean_curve
        return ece_mean_curve
