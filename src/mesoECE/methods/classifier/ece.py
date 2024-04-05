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
                                       define_masks_volume,
                                       correct_images_background)
from src.mesoECE.methods.utils_ece import (ece_label_selection,
                                           plot_ece_curves,
                                           export_curves_to_csv,
                                           interp_missing_points,
                                           define_superspels_curve_reg)


class ECE(AbstractMethod):
    def __init__(self, path: Path, ref_t: int, domain: str = None):
        super().__init__()

        self.thread_safe = False
        self.predicted_diagnosis = []
        self.path_ece = path
        self.ref_t = ref_t
        self.domain = domain

    def apply(self, patient: Patient, **kwargs):
        path_m_images = self.path_ece / 'ece_images'
        path_b_images = self.path_ece / 'benign_images'
        path_plot = self.path_ece / 'plots'
        path_ss_df = self.path_ece / 'superspels_df'

        os.makedirs(path_ss_df, exist_ok=True)
        os.makedirs(path_plot, exist_ok=True)
        os.makedirs(path_m_images, exist_ok=True)
        os.makedirs(path_b_images, exist_ok=True)

        try:
            print("\r", end='')
            print("ECE processing......", end="", flush=True)

            ss_mask, nifti_args = self.define_superspels_mask(patient)
            images_corrected = correct_images_background(patient)

            ss_mean_curves = self.define_superspels_curves(
                patient, images_corrected, ss_mask)

            index_270 = patient.time_points.index(270)

            ece_labels = ece_label_selection(
                index_270=index_270,
                ss_mean=ss_mean_curves)

            ece_mask, benign_mask = self.define_ece_mask(ece_labels, ss_mask)
            pleural_mask = patient.get_image(self.ref_t).masks[
                "pleural_region"].data.astype(np.int32)

            ece_vol = 0
            pleural_vol = 0

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
                             str(path_m_images / patient.get_image(
                                 self.ref_t).filename))
                    plot_ece_curves(curves=ss_mean_curves,
                                    time_points=patient.time_points,
                                    ece_mask=ece_mask[-1],
                                    filename=str(path_plot /
                                                 MRImage.resolve_name(
                                                     patient.id, self.ref_t,
                                                     "png")))
                elif self.domain == 'ORIG':
                    for i in range(len(ece_mask)):
                        nib.save(nib.Nifti1Image(ece_mask[i], **nifti_args[i]),
                                 str(path_m_images / patient.get_image(
                                     patient.time_points[i]).filename))
                    plot_ece_curves(curves=ss_mean_curves,
                                    time_points=patient.time_points,
                                    ece_mask=ece_mask[index_270],
                                    filename=str(path_plot /
                                                 MRImage.resolve_name(
                                                     patient.id, self.ref_t,
                                                     "png")))

                ss_ece_interp, time_points_interp = interp_missing_points(
                    ss_ece_curves=ss_ece_curves,
                    ss_mean_curves=ss_mean_curves,
                    time_points=patient.time_points)

                self.export_all_dfs(patient=patient,
                                    ss_mean_curves=ss_mean_curves,
                                    ss_ece_curves=ss_ece_curves,
                                    ss_ece_curves_interp=ss_ece_interp,
                                    time_points_interp=time_points_interp,
                                    path_superspels_df=path_ss_df)

            else:
                self.predicted_diagnosis.append(
                    [patient.id, patient.diagnosis, 0,
                     patient.subclass_diagnosis, patient.nodular,
                     ece_labels.__len__(), ece_vol])

            if self.domain == 'REG':
                nib.save(nib.Nifti1Image(benign_mask[-1], **nifti_args[-1]),
                         str(path_b_images / patient.get_image(
                             self.ref_t).filename))
            elif self.domain == 'ORIG':
                for i in range(len(ece_mask)):
                    nib.save(nib.Nifti1Image(benign_mask[i], **nifti_args[i]),
                             str(path_b_images / patient.get_image(
                                 patient.time_points[i]).filename))

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
        benign_mask = []

        for i in range(len(ss_mask)):
            temp_mask = np.copy(ss_mask[i])
            mask = np.isin(ss_mask[i], ece_labels)
            temp_mask[i][~mask] = 0
            ece_mask.append(temp_mask[i])

            ss_mask[i][mask] = 0
            benign_mask.append(ss_mask[i])
        return ece_mask, benign_mask

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

    def define_superspels_curves(self, patient: Patient, ss_mask,
                                 images_corrected):

        ss_mean_curve = np.zeros(
            (int(patient.get_image(self.ref_t).masks[
                     "supervoxels"].data.max()) + 1,
             patient.time_points.__len__()))

        if self.domain == 'REG':
            ss_mean_curve = define_superspels_curve_reg(patient,
                                                        images_corrected,
                                                        ss_mask,
                                                        self.ref_t)

        if self.domain == 'ORIG':

            for t in patient.time_points:
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
