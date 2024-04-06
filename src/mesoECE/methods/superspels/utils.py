import numpy as np
import pandas as pd
import skimage
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
import nibabel as nib
from src.mesoECE.data_structure import Patient, MRImage


def plot_curves(curve, mask, time_points,
                filename=None,
                title='ECE Curves', mean_plot=False):
    plt.figure(figsize=(10, 6))
    ece_labels = np.unique(mask[mask > 0])
    for label in ece_labels:
        plt.plot(time_points, curve[label],
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

    if mean_plot:
        fig = plt.figure(figsize=(10, 6))
        mean = np.mean(curve, axis=0)
        std = np.std(curve, axis=0)

        plt.plot(mean, label="Mean Curve", color="blue")
        plt.fill_between(range(len(mean)), mean - std, mean + std, color="blue",
                         alpha=0.2)

        plt.axvline(x=270, color='red')
        plt.xlabel("Time Points")
        plt.ylabel("")
        plt.title(f"Mean_{title}")
        plt.grid(True)
        if filename is not None:
            plt.savefig(f"mean_{filename}.png")
        else:
            plt.show()
        plt.close(fig)


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
    mean_intensity_curves = np.zeros(
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
            mean_intensity_curves[rp.label, patient.time_points.index(t)] = \
                img_in_bbox[lbl_in_bbox > 0].mean()
    return mean_intensity_curves


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


def define_superspels_mask(patient: Patient, domain, ref_t, sv_mask=None):
    ss_mask = []
    nifti_args = []
    if sv_mask is None:

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
    else:
        nifti_args = patient.get_image(ref_t).nifti_props
        ss_mask = sv_mask

    return ss_mask, nifti_args


def define_superspels_curves(patient: Patient, ss_mask,
                             images_corrected, domain, ref_t):
    mean_intensity_curves = np.zeros(
        (int(patient.get_image(ref_t).masks[
                 "supervoxels"].data.max()) + 1,
         patient.time_points.__len__()))

    if domain == 'REG':
        mean_intensity_curves = define_superspels_curve_reg(patient,
                                                            images_corrected,
                                                            ss_mask,
                                                            ref_t)

    if domain == 'ORIG':
        mean_intensity_curves = define_superspels_curve_orig(patient,
                                                             images_corrected,
                                                             ss_mask,
                                                             ref_t)

    return mean_intensity_curves


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