import numpy as np
import pandas as pd
import skimage
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
import nibabel as nib

from src.mesoECE.data_structure import Patient


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


def interp_curve_missing_time_points(curves, time_points):
    n_curves = curves.shape[0]
    ss_interp_curves = np.zeros((n_curves, 7))

    standard_time_points = [0, 40, 80, 180, 270, 540, 810]
    for i in range(n_curves):
        f_interp = interp1d(time_points, curves[i],
                            fill_value=curves[i, -1],
                            bounds_error=False)
        ss_interp_curves[i] = np.asarray(
            [f_interp(t) for t in standard_time_points[:1]])
    return ss_interp_curves, standard_time_points


def save_curves_and_interp_to_csv(patient, curves,
                                  path, curve_name):
    save_curves_to_csv(
        curves=curves,
        time_points=patient.time_points,
        filename=str(
            path /
            f'{curve_name}_{patient.id}.csv'))

    curves_interp, time_points_interp = interp_curve_missing_time_points(
        curves=curves,
        time_points=patient.time_points)

    save_curves_to_csv(
        curves=curves_interp,
        time_points=time_points_interp,
        filename=str(
            path /
            f'interp_{curve_name}_{patient.id}.csv'))


def define_superspels_curve_reg(patient: Patient,
                                images_corrected,
                                mask):
    mean_intensity_curves = np.zeros(
        (int(mask.max()) + 1,
         patient.time_points.__len__()))

    rps = skimage.measure.regionprops(mask)
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
                                 mask):
    mean_intensity_curves = np.zeros(
        (int(mask.data.max()) + 1,
         patient.time_points.__len__()))
    for t in patient.time_points:
        rps = skimage.measure.regionprops(mask[patient.time_points.index(t)])
        img = images_corrected[patient.time_points.index(t)]

        for rp in rps:
            slice_bbox = tuple(
                [slice(dim_start, dim_finish) for dim_start, dim_finish
                 in zip(rp.bbox[:3], rp.bbox[3:])])
            lbl_in_bbox = rp.image
            img_in_bbox = img[slice_bbox]

            mean_intensity_curves[rp.label, patient.time_points.index(t)] = \
                img_in_bbox[lbl_in_bbox > 0].mean()
    return mean_intensity_curves


def define_mean_intensity_curves(patient: Patient, mask,
                                 images_corrected, domain):
    mean_intensity_curves = None

    if domain == 'REG':
        mean_intensity_curves = define_superspels_curve_reg(
            patient=patient,
            images_corrected=images_corrected,
            mask=mask)

    if domain == 'ORIG':
        mean_intensity_curves = define_superspels_curve_orig(
            patient=patient,
            images_corrected=images_corrected,
            mask=mask)

    return mean_intensity_curves


def define_superspels_mask(patient: Patient, domain, ref_t):
    nifti_args = None
    ss_mask = None
    if domain == 'REG':
        nifti_args = patient.get_image(ref_t).nifti_props
        ss_mask = patient.get_image(ref_t).masks[
            "supervoxels"].data.astype(np.int32)
    elif domain == 'ORIG':
        ss_mask = []
        nifti_args = []
        for t in patient.time_points:
            nifti_args.append(
                patient.get_image(t).masks["supervoxels"].nifti_props)
            ss_mask.append(
                patient.get_image(t).masks["supervoxels"].data.astype(
                    np.int32))

    return ss_mask, nifti_args


def save_superspels_masks(mask, nifti_args, patient, domain, ref_t, path):
    if domain == 'REG':
        nib.save(nib.Nifti1Image(mask, **nifti_args),
                 str(path / patient.get_image(
                     ref_t).filename))
    elif domain == 'ORIG':
        for i in range(len(mask)):
            nib.save(nib.Nifti1Image(mask[i], **nifti_args[i]),
                     str(path / patient.get_image(
                         patient.time_points[i]).filename))
