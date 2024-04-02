import numpy as np
import os
import pandas as pd
import nibabel as nib
import skimage.measure
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d

from pathlib import Path

from src.mesoECE.data_structure.patient import Patient, MRImage
from src.mesoECE.methods import AbstractMethod


class ECE(AbstractMethod):
    def __init__(self, path: Path, ref_t: int, domain: str = None,
                 interpolate: bool = False):
        super().__init__()
        self.superspels_interp = None
        self.selected_superpels = None
        self.thread_safe = False
        self.superspels = None
        self.predicted_diagnosis = []
        self.path_ece = path
        self.ref_t = ref_t
        self.domain = domain

    def apply(self, patient: Patient, **kwargs):

        try:
            print("\r", end='')
            print("ECE processing......", end="", flush=True)
            supervoxel_mask = []
            nifti_args = []

            if self.domain == 'ORIG':
                self.superspels = np.zeros(
                    (int(patient.get_image(self.ref_t).masks[
                             "supervoxels"].data.max()),
                     patient.time_points.__len__()))

                for t in patient.time_points:
                    nifti_args.append(
                        patient.get_image(t).masks["supervoxels"].nifti_props)
                    supervoxel_mask.append(
                        patient.get_image(t).masks["supervoxels"].data.astype(
                            np.int32))
                    rps = skimage.measure.regionprops(supervoxel_mask[-1])

                    for rp in rps:
                        slice_bbox = tuple(
                            [slice(dim_start, dim_finish) for
                             dim_start, dim_finish in
                             zip(rp.bbox[:3], rp.bbox[3:])])
                        lbl_in_bbox = rp.image
                        img_in_bbox = patient.get_image(t).data[slice_bbox]

                        self.superspels[
                            rp.label - 1, patient.time_points.index(t)] = \
                            img_in_bbox[lbl_in_bbox > 0].mean()

            elif self.domain == 'REG':
                nifti_args.append(patient.get_image(self.ref_t).get_mask(
                    "supervoxels").nifti_props)
                supervoxel_mask.append(
                    patient.get_image(self.ref_t).get_mask(
                        "supervoxels").data.astype(np.int32))
                rps = skimage.measure.regionprops(supervoxel_mask[-1])
                self.superspels = np.zeros((len(rps), len(patient.time_points)))

                for rp in rps:
                    slice_bbox = tuple(
                        [slice(dim_start, dim_finish) for dim_start, dim_finish
                         in zip(rp.bbox[:3], rp.bbox[3:])])
                    lbl_in_bbox = rp.image

                    for t in patient.time_points:
                        img_in_bbox = patient.get_image(t).data[
                                          slice_bbox] - patient.background_otsu(
                            t)
                        self.superspels[
                            rp.label - 1, patient.time_points.index(t)] = \
                            img_in_bbox[
                                lbl_in_bbox > 0].mean()

            selected_labels = self._label_selection(
                patient.time_points.index(270))
            ece_mask = []

            for i in range(len(supervoxel_mask)):
                mask = np.isin(supervoxel_mask[i], selected_labels)
                supervoxel_mask[i][~mask] = 0
                ece_mask.append(supervoxel_mask[i])

            region_mask = patient.get_image(self.ref_t).masks[
                "pleural_region"].data.astype(np.int32)

            volume_mask = np.sum(region_mask)
            ece_sum = np.sum(ece_mask)
            if selected_labels and ece_sum > volume_mask * 0.0001:
                self.predicted_diagnosis.append(
                    [patient.id, patient.diagnosis, 1,
                     patient.subclass_diagnosis, patient.nodular,
                     selected_labels.__len__(), ece_sum])

                self.selected_superpels = np.zeros(
                    (len(selected_labels), len(patient.time_points)))

                for label in selected_labels:
                    self.selected_superpels[selected_labels.index(label)] = \
                        self.superspels[label]

                self.superspels_interp, time_points_interp = self.interpolate_missing_time_points(
                    time_points=patient.time_points)

                path_images = self.path_ece / 'images'
                path_plot_curves = self.path_ece / 'plots'
                path_selected_labels_df = self.path_ece / 'selected_labels_df'
                path_interp_df = self.path_ece / 'interp_df'

                os.makedirs(path_selected_labels_df, exist_ok=True)
                os.makedirs(path_plot_curves, exist_ok=True)
                os.makedirs(path_images, exist_ok=True)
                os.makedirs(path_interp_df, exist_ok=True)

                for i in range(len(ece_mask)):
                    if self.domain == 'REG':
                        nib.save(nib.Nifti1Image(ece_mask[i], **nifti_args[i]),
                                 str(path_images / patient.get_image(
                                     self.ref_t).filename))
                    elif self.domain == 'ORIG':
                        nib.save(nib.Nifti1Image(ece_mask[i], **nifti_args[i]),
                                 str(path_images / patient.get_image(
                                     patient.time_points[i]).filename))

                self.plot_ece_curves(selected_labels=selected_labels,
                                     time_points=patient.time_points,
                                     filename=str(
                                         path_plot_curves / MRImage.resolve_name(
                                             patient.id, self.ref_t,
                                             "png")))

                self.export_selected_labels_to_dataframe(
                    superspels=self.selected_superpels,
                    time_points=patient.time_points,
                    filename=str(
                        path_selected_labels_df / MRImage.resolve_name(
                            patient.id,
                            self.ref_t,
                            "csv")))

                self.export_selected_labels_to_dataframe(
                    superspels=self.superspels_interp,
                    time_points=time_points_interp,
                    filename=str(
                        path_interp_df / MRImage.resolve_name(patient.id,
                                                              self.ref_t,
                                                              "csv")))

            else:
                self.predicted_diagnosis.append(
                    [patient.id, patient.diagnosis, 0,
                     patient.subclass_diagnosis, patient.nodular,
                     selected_labels.__len__(), ece_sum])

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

    def _label_selection(self, index_270):
        valid_indices = []
        for i in range(self.superspels.shape[0]):
            peak_index = np.argmax(self.superspels[i])

            if 0 < peak_index <= index_270:
                is_increasing = np.all(
                    self.superspels[i, :peak_index - 1] < self.superspels[i,
                                                          1:peak_index])
                is_decreasing = np.all(
                    self.superspels[i, peak_index:-1] > self.superspels[i,
                                                        peak_index + 1:])
                if is_increasing and is_decreasing:
                    valid_indices.append(i)
        return valid_indices

    def interpolate_missing_time_points(self, time_points):
        n_curves = self.selected_superpels.shape[0]
        superspels_interp = np.zeros((n_curves, 7))

        standard_time_points = [0, 40, 80, 180, 270, 540, 810]
        for i in range(n_curves):
            f_interp = interp1d(time_points, self.selected_superpels[i],
                                fill_value=self.superspels[i, -1],
                                bounds_error=False)
            superspels_interp[i] = np.asarray(
                [f_interp(t) for t in standard_time_points[:1]])
        return superspels_interp, standard_time_points

    def plot_ece_curves(self, selected_labels, time_points, filename=None,
                        title='ECE Curves'):
        plt.figure(figsize=(10, 6))
        for label in selected_labels:
            plt.plot(time_points, self.superspels[label],
                     label=f'Label {label}')

        plt.xlabel('Time Points')
        plt.ylabel('Mean Intensity')
        # Add a red vertical line at x=3
        plt.axvline(x=270, color='red')
        plt.title(title)
        # plt.legend()
        plt.grid(True)

        if filename:
            plt.savefig(filename)  # Save the plot as a PNG file
        plt.close('all')

    @staticmethod
    def export_selected_labels_to_dataframe(superspels, time_points,
                                            filename='selected_superspels.csv'):
        """
        Export data and corresponding time points as a Pandas DataFrame and save it to a CSV file.
        """

        column_names_str = list(map(str, time_points))
        df = pd.DataFrame(superspels, columns=column_names_str)
        df.to_csv(filename, index=False)
