import numpy as np
import os
import pandas as pd
import nibabel as nib
import scipy
import skimage.measure
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d

from pathlib import Path

from src.mesoECE.data_structure.patient import Patient, MRImage
from src.mesoECE.methods import AbstractMethod
from src.mesoECE.methods.utils import (correct_image_background,
                                       export_curves_to_csv)

from src.mesoECE.methods.utils import define_masks_volume


def ece_label_selection(index_270, ss_mean):
    ece_labels = []
    for i in range(ss_mean.shape[0]):
        peak_index = np.argmax(ss_mean[i])

        if 0 < peak_index <= index_270:
            is_increasing = np.all(
                ss_mean[i, :peak_index - 1] < ss_mean[i, 1:peak_index])
            is_decreasing = np.all(
                ss_mean[i, peak_index:-1] > ss_mean[i, peak_index + 1:])
            pre_contrast = np.all(ss_mean[i, 0] < ss_mean[i, 1:])
            if is_increasing and is_decreasing and pre_contrast:
                ece_labels.append(i)
    return ece_labels


class ECE(AbstractMethod):
    def __init__(self, path: Path, ref_t: int, domain: str = None,
                 interpolate: bool = False):
        super().__init__()

        self.thread_safe = False
        self.predicted_diagnosis = []
        self.path_ece = path
        self.ref_t = ref_t
        self.domain = domain

    def apply(self, patient: Patient, **kwargs):
        path_images = self.path_ece / 'images'
        path_plot = self.path_ece / 'plots'
        path_ss_df = self.path_ece / 'superspels_df'

        os.makedirs(path_ss_df, exist_ok=True)
        os.makedirs(path_plot, exist_ok=True)
        os.makedirs(path_images, exist_ok=True)

        try:
            print("\r", end='')
            print("ECE processing......", end="", flush=True)

            ss_mask, nifti_args = self.define_superspels_mask(patient)

            ss_mean_curves = self.define_superspels_curves(
                patient, ss_mask, nifti_args)

            index_270 = patient.time_points.index(270)

            ece_labels = self.ece_label_selection(
                index_270=index_270,
                ss_mean=ss_mean_curves)

            ece_mask = self.define_ece_mask(ece_labels, ss_mask)
            pleural_mask = patient.get_image(self.ref_t).masks[
                "pleural_region"].data.astype(np.int32)

            if self.domain == 'REG':

                pleural_vol = define_masks_volume(mask=pleural_mask)
                ece_vol = define_masks_volume(mask=ece_mask[-1])
            elif self.domain == 'ORIG':
                pleural_vol = define_masks_volume(mask=pleural_mask)
                ece_vol = define_masks_volume(mask=ece_mask[index_270])

            if ece_labels and ece_vol > pleural_vol * 0.0001:
                self.predicted_diagnosis.append(
                    [patient.id, patient.diagnosis, 1,
                     patient.subclass_diagnosis, patient.nodular,
                     ece_labels.__len__(), ece_vol])

                ss_ece_curves = self.define_ece_curves(
                    len_time_points=len(patient.time_points),
                    ss_mean_curves=ss_mean_curves,
                    ece_labels=ece_labels)

                if self.domain == 'REG':
                    nib.save(nib.Nifti1Image(ece_mask[-1], **nifti_args[-1]),
                             str(path_images / patient.get_image(
                                 self.ref_t).filename))
                elif self.domain == 'ORIG':
                    for i in range(len(ece_mask)):
                        nib.save(nib.Nifti1Image(ece_mask[i], **nifti_args[i]),
                                 str(path_images / patient.get_image(
                                     patient.time_points[i]).filename))

                self.plot_ece_curves(ece_labels=ece_labels,
                                     time_points=patient.time_points,
                                     ss_mean_curves=ss_mean_curves,
                                     filename=str(path_plot /
                                                  MRImage.resolve_name(
                                                      patient.id, self.ref_t,
                                                      "png")))

                ss_ece_interp, time_points_interp = self.interp_missing_points(
                    ss_ece_curves=ss_ece_curves,
                    ss_mean_curves=ss_mean_curves,
                    time_points=patient.time_points)

                self.export_all_dfs(patient, ss_mean_curves, time_points_interp,
                                    path_ss_df)

            else:
                self.predicted_diagnosis.append(
                    [patient.id, patient.diagnosis, 0,
                     patient.subclass_diagnosis, patient.nodular,
                     ece_labels.__len__(), ece_vol])

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
    def define_ece_curves(len_time_points, ss_mean_curves, ece_labels):
        ss_ece_curves = np.zeros(
            (len(ece_labels), len_time_points))

        for l in ece_labels:
            ss_ece_curves[ece_labels.index(l)] = ss_mean_curves[l]

        return ss_ece_curves

    @staticmethod
    def define_ece_mask(ece_labels, ss_mask):
        ece_mask = []

        for i in range(len(ss_mask)):
            mask = np.isin(ss_mask[i], ece_labels)
            ss_mask[i][~mask] = 0
            ece_mask.append(ss_mask[i])
        return ece_mask

    def define_superspels_mask(self, patient: Patient):
        ss_mask = []
        nifti_args = []

        if self.domain == 'REG':
            nifti_args.append(patient.get_image(self.ref_t).nifti_props)
            ss_mask.append(patient.get_image(self.ref_t).masks[
                               "supervoxels"].data.astype(np.int32))

        elif self.domain == 'ORIG':
            for t in patient.time_points:
                nifti_args.append(
                    patient.get_image(t).masks["supervoxels"].nifti_props)
                ss_mask.append(
                    patient.get_image(t).masks["supervoxels"].data.astype(
                        np.int32))

        return ss_mask, nifti_args

    def define_superspels_curves(self, patient: Patient, ss_mask, nifti_args):

        ss_mean_curve = np.zeros(
            (int(patient.get_image(self.ref_t).masks[
                     "supervoxels"].data.max()) + 1,
             patient.time_points.__len__()))

        if self.domain == 'REG':
            images = []
            for t in patient.time_points:
                images = correct_image_background(patient, t)
            rps = skimage.measure.regionprops(ss_mask[-1])
            for rp in rps:
                slice_bbox = tuple(
                    [slice(dim_start, dim_finish) for dim_start, dim_finish in
                     zip(rp.bbox[:3], rp.bbox[3:])])
                lbl_in_bbox = rp.image

                for t in patient.time_points:
                    img = images[patient.time_points.index(t)]
                    img_in_bbox = img[slice_bbox]
                    ss_mean_curve[rp.label - 1, patient.time_points.index(t)] = \
                        img_in_bbox[lbl_in_bbox > 0].mean()

        if self.domain == 'ORIG':

            for t in patient.time_points:
                nifti_args.append(
                    patient.get_image(t).masks["supervoxels"].nifti_props)
                ss_mask.append(
                    patient.get_image(t).masks["supervoxels"].data.astype(
                        np.int32))
                rps = skimage.measure.regionprops(ss_mask[-1])
                img = correct_image_background(patient, t)

                for rp in rps:
                    slice_bbox = tuple(
                        [slice(dim_start, dim_finish) for dim_start, dim_finish
                         in zip(rp.bbox[:3], rp.bbox[3:])])
                    lbl_in_bbox = rp.image
                    img_in_bbox = img[slice_bbox]

                    ss_mean_curve[rp.label, patient.time_points.index(t)] = \
                        img_in_bbox[lbl_in_bbox > 0].mean()

        return ss_mean_curve

    @staticmethod
    def interp_missing_points(ss_ece_curves,
                              ss_mean_curves, time_points):
        n_curves = ss_ece_curves.shape[0]
        ss_interp_curves = np.zeros((n_curves, 7))

        standard_time_points = [0, 40, 80, 180, 270, 540, 810]
        for i in range(n_curves):
            f_interp = interp1d(time_points, ss_ece_curves[i],
                                fill_value=ss_mean_curves[i, -1],
                                bounds_error=False)
            ss_interp_curves[i] = np.asarray(
                [f_interp(t) for t in standard_time_points[:1]])
        return ss_interp_curves, standard_time_points

    @staticmethod
    def plot_ece_curves(ss_mean_curves, ece_labels, time_points,
                        filename=None,
                        title='ECE Curves'):
        plt.figure(figsize=(10, 6))
        for label in ece_labels:
            plt.plot(time_points, ss_mean_curves[label],
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

    def export_all_dfs(self, patient: Patient, ss_mean_curves, ss_ece_curves,
                       ss_ece_curves_interp, time_points_interp,
                       path_superspels_df: Path):
        export_curves_to_csv(
            curves=ss_ece_curves,
            time_points=patient.time_points,
            filename=str(
                path_superspels_df /
                f'ece_{MRImage.resolve_name(patient.id,
                                            self.ref_t,
                                            "csv")}'))

        export_curves_to_csv(
            curves=ss_ece_curves_interp,
            time_points=time_points_interp,
            filename=str(
                path_superspels_df /
                f'intep_ece_{MRImage.resolve_name(patient.id,
                                                  self.ref_t,
                                                  "csv")}'))

        export_curves_to_csv(
            curves=ss_mean_curves,
            time_points=patient.time_points,
            filename=str(
                path_superspels_df /
                f'all_{MRImage.resolve_name(patient.id,
                                            self.ref_t,
                                            "csv")}'))
