import os
import shutil
import nibabel as nib
import numpy as np
import skimage
from skimage.segmentation import slic
from pathlib import Path

from src.mesoECE.data_structure.patient import Patient, Diagnosis
from src.mesoECE.methods import AbstractMethod


class DividingECE(AbstractMethod):
    def __init__(self, path: Path, ref_t: int, n_segments: int, compactness: float, percentage: int,
                 method: str = None,
                 domain: str = None):
        super().__init__()
        self.path_supervoxels = path
        self.domain = domain
        self.ref_t = ref_t
        self.n_segments = n_segments
        self.compactness = compactness
        self.thread_safe = False
        self.predicted_diagnosis = []
        self.volume_mask = 0
        self.current_diagosis = None
        self.method = method
        self.percentage = percentage

    def apply(self, patient: Patient, **kwargs):
        try:

            nifti_args = patient.get_image(self.ref_t).masks["pleural_region"].nifti_props
            img_header = nifti_args["header"]
            pleural_region_mask = patient.get_image(self.ref_t).masks["pleural_region"].data.astype(np.int32)


            img = patient.images(self.ref_t).data
            self.fluid_volume_mask  = np.sum(pleural_region_mask)

            path_img = patient.get_image(self.ref_t).path / patient.get_image(self.ref_t).filename
            path_mask = patient.get_image(self.ref_t).path_masks['pleural_region'] / patient.get_image(
                self.ref_t).filename

            if self.method == 'SLIC':

                self.ece_region_dividing_SLIC(patient, img, pleural_region_mask)
            elif self.method == 'DISF':
                os.makedirs(self.path_supervoxels / 'TempDISF', exist_ok=True)
                os.makedirs(self.path_supervoxels / 'TempDISF2', exist_ok=True)
                self.ece_region_dividing_DISF(patient, pleural_region_mask, path_img, path_mask, nifti_args, counter=0)
                shutil.rmtree(self.path_supervoxels / 'TempDISF')
                shutil.rmtree(self.path_supervoxels / 'TempDISF2')
            elif self.method == 'SICLE':
                self.ece_region_dividing_SICLE(patient, img, pleural_region_mask)

            # combine the masks in supervoxel_mask, so that the voxels in the same supervoxel have the same label
            supervoxel_mask = np.zeros_like(pleural_region_mask)
            for i, supervoxel in enumerate(patient.supervoxels_masks):
                supervoxel_mask[supervoxel != 0] = i + 1
            mask_temp = np.zeros_like(supervoxel_mask)
            mask_temp[supervoxel_mask > 0] = 1

            #classification
            if patient.supervoxels_masks.__len__() == 0 or np.sum(mask_temp) < self.fluid_volume_mask  * 0.0001:
                print("No supervoxels found!")
                self.predicted_diagnosis.append(
                    [patient.id, patient.diagnosis, Diagnosis.NON_MALIGNANT, patient.supervoxels_masks.__len__(), 0])
                print(
                    f"[{patient.id}, {patient.diagnosis}, {Diagnosis.NON_MALIGNANT}, {patient.supervoxels_masks.__len__()},O]")
            else:
                print(patient.supervoxels_masks.__len__(), "supervoxels found!")
                self.predicted_diagnosis.append(
                    [patient.id, patient.diagnosis, Diagnosis.MALIGNANT, patient.supervoxels_masks.__len__(),
                     np.sum(mask_temp)])
                print(
                    f"[{patient.id}, {patient.diagnosis}, {Diagnosis.MALIGNANT}, {patient.supervoxels_masks.__len__()}, {np.sum(mask_temp)}]")

            # while all(self.ece_selection(patient, mask) == Diagnosis.NON_MALIGNANT for mask in masks):
            # plt.imshow(pleural_region_mask[:, :, 100])
            # plt.show()
            nib.save(nib.Nifti1Image(supervoxel_mask.astype(np.int32), **nifti_args),
                     str(self.path_supervoxels / patient.get_image(self.ref_t).filename))

            patient.path_masks['supervoxels'] = self.path_supervoxels

        except:
            print("Error in patient:", patient.id)
        new_patient = Patient(path=patient.path,
                              path_masks=patient.path_masks,
                              id=patient.id,
                              diagnosis=patient.diagnosis,
                              subclass_diagnosis=patient.subclass_diagnosis,
                              nodular=patient.nodular)
        return new_patient

    def ece_region_dividing_SLIC(self, patient: Patient, img, mask):
        # if :
        if np.sum(mask) >= self.fluid_volume_mask * self.percentage:
            print("\r", end='')
            print("Dividing...", end="", flush=True)

            supervoxel_mask = slic(img, mask=mask,
                                   channel_axis=None,
                                   # sigma = 1,
                                   compactness=self.compactness, n_segments=self.n_segments)
            rps = skimage.measure.regionprops(supervoxel_mask)

            for rp in rps:
                mask_temp = np.zeros_like(mask)

                slice_bbox = tuple(
                    [slice(dim_start, dim_finish) for dim_start, dim_finish in zip(rp.bbox[:3], rp.bbox[3:])])
                lbl_in_bbox = rp.image
                mask_temp[slice_bbox] = lbl_in_bbox

                self.ece_region_dividing_SLIC(patient, img, mask_temp)
        elif self.ece_selection(patient, mask) == 1 and np.sum(mask) < self.volume_mask * self.percentage:
            return patient.supervoxels_masks.append(mask)

    def ece_region_dividing_DISF(self, patient: Patient, mask, path_img, path_mask, nifti_args, counter=0):
        # if self.ece_selection(patient, mask) == 0 and np.sum(mask) > self.volume_mask * 0.22:
        # plt.imshow(mask[:, :, 100])
        #
        # plt.show()
        if self.ece_selection(patient, mask) == 0 and np.sum(mask) >= self.volume_mask * 0.25:
            print("\r", end='')
            print("Dividing...", end="", flush=True)

            path_out_mask = self.path_supervoxels / 'TempDISF' / f"{counter}.nii.gz"

            cmd = f"/app/data/ift/bin/iftDISF {path_img} {self.n_segments * 30} {self.n_segments} {path_out_mask} {path_mask}"

            print("Running command: " + cmd)

            if (os.system(cmd) == -1):
                print("Error at iftDISF. Please compile the program iftDISF")
                exit(-1)

            supervoxel_mask_nib = nib.load(str(path_out_mask))
            supervoxel_mask = supervoxel_mask_nib.get_fdata().astype(np.int32)

            rps = skimage.measure.regionprops(supervoxel_mask)

            for rp in rps:
                mask_temp = np.zeros_like(mask)
                slice_bbox = tuple(
                    [slice(dim_start, dim_finish) for dim_start, dim_finish in zip(rp.bbox[:3], rp.bbox[3:])])
                lbl_in_bbox = rp.image
                mask_temp[slice_bbox] = lbl_in_bbox
                path_out_mask_temp = self.path_supervoxels / 'TempDISF2' / f"{counter}.nii.gz"
                nib.save(nib.Nifti1Image(mask_temp.astype(np.int32), **nifti_args),
                         str(path_out_mask_temp))
                counter = counter + 1
                self.ece_region_dividing_DISF(patient, mask_temp, path_img, path_out_mask_temp, nifti_args,
                                              counter=counter)

        # elif self.ece_selection(patient, mask) == 1:
        #     self.current_diagosis = [patient.id, patient.diagnosis, Diagnosis.MALIGNANT]
        #     return patient.supervoxels_masks.append(mask)
        # elif self.ece_selection(patient, mask) == 0 and np.sum(mask) < self.volume_mask * 0.25:
        #     self.current_diagosis = [patient.id, patient.diagnosis, Diagnosis.NON_MALIGNANT]

    # TODO: SICLE
    def ece_region_dividing_SICLE(self, patient: Patient, img, mask):
        # if self.ece_selection(patient, mask) == 0 and np.sum(mask) > self.volume_mask * 0.22:
        if self.ece_selection(patient, mask) == 0 and np.sum(mask) >= self.volume_mask * 0.05:
            print("\r", end='')
            print("Dividing...", end="", flush=True)

            supervoxel_mask = slic(img, mask=mask,
                                   channel_axis=None,
                                   compactness=self.compactness, n_segments=self.n_segments)
            rps = skimage.measure.regionprops(supervoxel_mask)

            for rp in rps:
                mask_temp = np.zeros_like(mask)

                slice_bbox = tuple(
                    [slice(dim_start, dim_finish) for dim_start, dim_finish in zip(rp.bbox[:3], rp.bbox[3:])])
                lbl_in_bbox = rp.image
                mask_temp[slice_bbox] = lbl_in_bbox

                self.ece_region_dividing(patient, img, mask_temp)
        elif self.ece_selection(patient, mask) == 1:
            self.current_diagosis = [patient.id, patient.diagnosis, Diagnosis.MALIGNANT]
            return patient.supervoxels_masks.append(mask)
        elif self.ece_selection(patient, mask) == 0 and np.sum(mask) < self.volume_mask * 0.05:
            self.current_diagosis = [patient.id, patient.diagnosis, Diagnosis.NON_MALIGNANT]

    def ece_selection(self, patient: Patient, supervoxel_mask):
        nifti_args = []
        # TODO: ORIG
        # if self.domain == 'ORIG':
        # self.superspels = np.zeros(
        #     (int(supervoxel_mask.max()), patient.time_points.__len__()))
        #
        # for t in patient.time_points:
        #     nifti_args.append(patient.images(t).masks["supervoxels"].nifti_props)
        #     supervoxel_mask.append(patient.images(t).masks["supervoxels"].data.astype(np.int32))
        #     rps = skimage.measure.regionprops(supervoxel_mask[-1])
        #
        #     for rp in rps:
        #         slice_bbox = tuple(
        #             [slice(dim_start, dim_finish) for dim_start, dim_finish in zip(rp.bbox[:3], rp.bbox[3:])])
        #         lbl_in_bbox = rp.image
        #         img_in_bbox = patient.images(t).data[slice_bbox]
        #
        #         self.superspels[rp.label - 1, patient.time_points.index(t)] = img_in_bbox[lbl_in_bbox > 0].mean()

        # elif self.domain == 'REG':
        rps = skimage.measure.regionprops(supervoxel_mask)
        superspels = np.zeros((len(rps), len(patient.time_points)))

        for rp in rps:
            slice_bbox = tuple(
                [slice(dim_start, dim_finish) for dim_start, dim_finish in zip(rp.bbox[:3], rp.bbox[3:])])
            lbl_in_bbox = rp.image

            for t in patient.time_points:
                img_in_bbox = patient.images(t).data[slice_bbox] - - patient.background_otsu(t)
                superspels[rp.label - 1, patient.time_points.index(t)] = img_in_bbox[lbl_in_bbox > 0].mean()

        selected_labels = self._label_selection(superspels=superspels, index_270=patient.time_points.index(270))

        if selected_labels:
            predicted_diagnosis = 1
            # ece_mask = []
            #
            # for i in range(len(supervoxel_mask)):
            #     mask = np.isin(supervoxel_mask[i], selected_labels)
            #     supervoxel_mask[i][~mask] = 0
            #     ece_mask.append(supervoxel_mask[i])
            #
            # self.selected_superpels = np.zeros((len(selected_labels), len(patient.time_points)))
            #
            # for label in selected_labels:
            #     self.selected_superpels[selected_labels.index(label)] = self.superspels[label]
        else:
            predicted_diagnosis = 0

        return predicted_diagnosis

    @staticmethod
    def _label_selection(superspels, index_270):
        valid_indices = []

        for i in range(superspels.shape[0]):
            peak_index = np.argmax(superspels[i])

            if 0 < peak_index <= index_270:
                is_increasing = np.all(superspels[i, :peak_index - 1] < superspels[i, 1:peak_index])
                is_decreasing = np.all(superspels[i, peak_index:-1] > superspels[i, peak_index + 1:])
                if is_increasing and is_decreasing:
                    # print("superspelsECE:", superspels)
                    valid_indices.append(i)
        return valid_indices

    def results(self):
        return self.predicted_diagnosis