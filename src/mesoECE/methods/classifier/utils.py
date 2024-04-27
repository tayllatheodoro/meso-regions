import numpy as np
import skimage


def superspels_labels_with_ece(index_270:int, mean_intensity_curve: np.ndarray):
    ece_labels = []
    for i in range(mean_intensity_curve.shape[0]):
        p_index = np.argmax(mean_intensity_curve[i])

        if 0 < p_index <= index_270:
            is_increasing = np.all(
                mean_intensity_curve[i, :p_index - 1] <
                mean_intensity_curve[i, 1:p_index])

            is_decreasing = np.all(
                mean_intensity_curve[i, p_index:-1] >
                mean_intensity_curve[i, p_index + 1:])

            pre_contrast = np.all(mean_intensity_curve[i, 0] <
                                  mean_intensity_curve[i, 1:])

            if is_increasing and is_decreasing and pre_contrast:
                ece_labels.append(i)
    return ece_labels


def define_ece_curves(mean_intensity_curves: np.ndarray, ece_labels: list):
    ece_curves = np.zeros_like(mean_intensity_curves)

    for label in ece_labels:
        ece_curves[label] = mean_intensity_curves[label]

    benign_curves = np.delete(mean_intensity_curves, ece_labels, axis=0)
    return ece_curves, benign_curves


def define_ece_mask(ece_labels:list, ss_mask: np.ndarray):
    ece_mask = np.zeros_like(ss_mask)
    rps = skimage.measure.regionprops(ss_mask)
    for rp in rps:
        if (rp.label in ece_labels) and np.sum(
                ss_mask[ss_mask == rp.label]) > 0:
            ece_mask[ss_mask == rp.label] = rp.label


    return ece_mask
