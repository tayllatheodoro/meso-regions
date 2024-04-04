import numpy as np
import os
import pandas as pd
import nibabel as nib
import skimage.measure
from matplotlib import pyplot as plt
from scipy.ndimage import uniform_filter
from pathlib import Path

from src.mesoECE.data_structure import MRImage
from src.mesoECE.data_structure.patient import Patient, Diagnosis
from src.mesoECE.methods import AbstractMethod
from src.mesoECE.methods.classifier.ece import correct_image_background, \
    export_curves_to_csv


class FullECE(AbstractMethod):
    def __init__(self, path: Path, ref_t: int, filter_size: int = 5,
                 with_mask: bool = True):
        super().__init__()
        self.thread_safe = True
        self.predicted_diagnosis = []
        self.path_ece = path
        self.ref_t = ref_t
        self.filter_size = filter_size
        self.with_mask = with_mask

    def apply(self, patient: Patient, **kwargs):

        try:
            path_images = self.path_ece / 'images'
            path_plot = self.path_ece / 'plots'
            path_df = self.path_ece / 'superspels_df'

            os.makedirs(path_images, exist_ok=True)
            os.makedirs(path_df, exist_ok=True)
            os.makedirs(path_plot, exist_ok=True)

            nifti_args = patient.get_image(self.ref_t).nifti_props
            pleural_mask = patient.get_image(self.ref_t).masks[
                "pleural_region"].data.astype(np.int32)

            images = self.define_images_filtered(patient, pleural_mask)
            ece_mask = self.define_ece_mask(patient, images)

            ece_labeled = skimage.measure.label(ece_mask)

            pleural_vol, ece_vol = self.define_masks_volume(
                pleural_mask=pleural_mask,
                ece_mask=ece_mask)

            if ece_vol > pleural_vol * 0.0001:
                self.predicted_diagnosis.append(
                    [patient.id, patient.diagnosis, 1,
                     patient.subclass_diagnosis, patient.nodular,
                     ece_vol, ece_vol])

                nib.save(nib.Nifti1Image(ece_labeled, **nifti_args),
                         str(path_images / patient.get_image(
                             self.ref_t).filename))
                ece_curves = self.define_ece_curves(patient, ece_labeled)
                export_curves_to_csv(
                    curves=ece_curves,
                    time_points=patient.time_points,
                    filename=str(
                        path_df /
                        f'all_{MRImage.resolve_name(patient.id,
                                                    self.ref_t,
                                                    "csv")}'))
                self.plot_ece_curves(curves=ece_curves,
                                     ece_mask=ece_labeled,
                                     time_points=patient.time_points,
                                     filename=str(path_plot /
                                                  MRImage.resolve_name(
                                                      patient.id, self.ref_t,
                                                      "png")))

            else:
                self.predicted_diagnosis.append(
                    [patient.id, patient.diagnosis, 0,
                     patient.subclass_diagnosis, patient.nodular,
                     ece_vol, ece_vol])

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

    @staticmethod
    def define_masks_volume(pleural_mask, ece_mask):
        pleural_volume = np.sum(pleural_mask)
        ece_volume = np.sum(ece_mask)
        return pleural_volume, ece_volume

    def define_images_filtered(self, patient, pleural_mask):
        images = []
        for t in patient.time_points:
            # image mean filter using skitmage
            img_corrected = correct_image_background(patient, t)
            img_filtered = uniform_filter(img_corrected,
                                          size=self.filter_size)
            if self.with_mask:
                img_filtered[pleural_mask == 0] = 0
            images.append(img_filtered)
        return images

    def define_ece_mask(self, patient, images):
        ece_mask = np.ones_like(patient.get_image(self.ref_t).data.shape)
        peal_index = patient.time_points.index(270)
        for i, t in enumerate(patient.time_points):
            if i < peal_index:
                ece_mask = np.logical_and(ece_mask, images[i] < images[i + 1])
            elif i > peal_index:
                ece_mask = np.logical_and(ece_mask, images[i] < images[i - 1])
        return ece_mask

    def define_ece_curves(self, patient: Patient, ece_mask):

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
                ece_mean_curve[rp.label - 1, patient.time_points.index(t)] = \
                    img_in_bbox[lbl_in_bbox > 0]
        return ece_mean_curve

    @staticmethod
    def plot_ece_curves(curves, ece_mask, time_points,
                        filename=None,
                        title='ECE Curves'):
        plt.figure(figsize=(10, 6))
        ece_labels = np.unique(ece_mask)
        for label in ece_labels:
            plt.plot(time_points, curves[label],
                     label=f'Label {label}')

        plt.xlabel('Time Points')
        plt.ylabel('Mean Intensity')
        # Add a red vertical line at x=3
        plt.axvline(x=270, color='red')
        plt.title(title)
        plt.grid(True)

        if filename:
            plt.savefig(filename)  # Save the plot as a PNG file

        else:
            plt.show()
        plt.close('all')
