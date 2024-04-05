import numpy as np
import pandas as pd
import skimage
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d

from src.mesoECE.data_structure import Patient
from src.mesoECE.methods.utils import correct_image_background


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


def export_curves_to_csv(curves, time_points, filename=None):
    column_names_str = list(map(str, time_points))
    df = pd.DataFrame(curves, columns=column_names_str)
    df.to_csv(filename, index=False)


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
