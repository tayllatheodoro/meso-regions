import numpy as np
import pandas as pd
import skimage
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
import nibabel as nib
from src.mesoECE.data_structure import Patient, MRImage


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


def save_curves_to_csv(curves, time_points, filename=None):
    column_names_str = list(map(str, time_points))
    df = pd.DataFrame(curves, columns=column_names_str)
    df.to_csv(filename, index=False)


def interp_curve_missing_time_points(ss_curves, time_points):
    n_curves = ss_curves.shape[0]
    ss_interp_curves = np.zeros((n_curves, 7))

    standard_time_points = [0, 40, 80, 180, 270, 540, 810]
    for i in range(n_curves):
        f_interp = interp1d(time_points, ss_curves[i],
                            fill_value=ss_curves[i, -1],
                            bounds_error=False)
        ss_interp_curves[i] = np.asarray(
            [f_interp(t) for t in standard_time_points[:1]])
    return ss_interp_curves, standard_time_points


def save_curves_and_interp_to_csv(patient, curves, ref_t,
                                  path, curve_name):
    save_curves_to_csv(
        curves=curves,
        time_points=patient.time_points,
        filename=str(
            path /
            f'{curve_name}_{MRImage.resolve_name(patient.id,
                                                 ref_t,
                                                 "csv")}'))

    curves_interp, time_points_interp = interp_curve_missing_time_points(
        ss_curves=curves,
        time_points=patient.time_points)

    save_curves_to_csv(
        curves=curves_interp,
        time_points=time_points_interp,
        filename=str(
            path /
            f'intep_{curve_name}_{MRImage.resolve_name(patient.id,
                                                       ref_t,
                                                       "csv")}'))


def define_superspels_curve_reg(patient: Patient,
                                images_corrected,
                                ss_mask,
                                ref_t):
    ss_mean_curve = np.zeros(
        (int(patient.get_image(ref_t).masks[
                 "supervoxels"].data.max()) + 1,
         patient.time_points.__len__()))
    rps = skimage.measure.regionprops(ss_mask[-1])
    for rp in rps:
        slice_bbox = tuple(
            [slice(dim_start, dim_finish) for dim_start, dim_finish in
             zip(rp.bbox[:3], rp.bbox[3:])])
        lbl_in_bbox = rp.image

        for t in patient.time_points:
            img = images_corrected[patient.time_points.index(t)]
            img_in_bbox = img[slice_bbox]
            ss_mean_curve[rp.label - 1, patient.time_points.index(t)] = \
                img_in_bbox[lbl_in_bbox > 0].mean()
    return ss_mean_curve


def define_superspels_curve_orig(patient: Patient,
                                 images_corrected,
                                 ss_mask,
                                 ref_t):
    ss_mean_curve = np.zeros(
        (int(patient.get_image(ref_t).masks[
                 "supervoxels"].data.max()) + 1,
         patient.time_points.__len__()))
    for t in patient.time_points:
        rps = skimage.measure.regionprops(ss_mask[-1])
        img = images_corrected[patient.time_points.index(t)]

        for rp in rps:
            slice_bbox = tuple(
                [slice(dim_start, dim_finish) for dim_start, dim_finish
                 in zip(rp.bbox[:3], rp.bbox[3:])])
            lbl_in_bbox = rp.image
            img_in_bbox = img[slice_bbox]

            ss_mean_curve[rp.label, patient.time_points.index(t)] = \
                img_in_bbox[lbl_in_bbox > 0].mean()
    return ss_mean_curve


def define_superspels_mask(patient: Patient, domain, ref_t):
    ss_mask = []
    nifti_args = []

    if domain == 'REG':
        nifti_args.append(patient.get_image(ref_t).nifti_props)
        ss_mask.append(patient.get_image(ref_t).masks[
                           "supervoxels"].data.astype(np.int32))

    elif domain == 'ORIG':
        for t in patient.time_points:
            nifti_args.append(
                patient.get_image(t).masks["supervoxels"].nifti_props)
            ss_mask.append(
                patient.get_image(t).masks["supervoxels"].data.astype(
                    np.int32))

    return ss_mask, nifti_args


def define_superspels_curves(patient: Patient, ss_mask,
                             images_corrected, domain, ref_t):
    ss_mean_curve = np.zeros(
        (int(patient.get_image(ref_t).masks[
                 "supervoxels"].data.max()) + 1,
         patient.time_points.__len__()))

    if domain == 'REG':
        ss_mean_curve = define_superspels_curve_reg(patient,
                                                    images_corrected,
                                                    ss_mask,
                                                    ref_t)

    if domain == 'ORIG':
        ss_mean_curve = define_superspels_curve_orig(patient,
                                                     images_corrected,
                                                     ss_mask,
                                                     ref_t)

    return ss_mean_curve


def save_superspels_masks(ss_mask, nifti_args, patient, domain, ref_t, path):
    if domain == 'REG':
        nib.save(nib.Nifti1Image(ss_mask[-1], **nifti_args[-1]),
                 str(path / patient.get_image(
                     ref_t).filename))
    elif domain == 'ORIG':
        for i in range(len(ss_mask)):
            nib.save(nib.Nifti1Image(ss_mask[i], **nifti_args[i]),
                     str(path / patient.get_image(
                         patient.time_points[i]).filename))


def define_ece_curves(len_time_points, ss_mean_curves, ece_labels):
    ss_ece_curves = np.zeros(
        (len(ece_labels), len_time_points))

    for l in ece_labels:
        ss_ece_curves[ece_labels.index(l)] = ss_mean_curves[l]

    return ss_ece_curves


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
