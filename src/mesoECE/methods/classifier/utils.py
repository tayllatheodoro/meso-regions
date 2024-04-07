import numpy as np


def superspels_labels_with_ece(index_270, mean_intensity_curve):
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


def define_ece_curves(len_time_points, mean_intensity_curves, ece_labels):
    ece_curves = np.zeros(
        (len(ece_labels), len_time_points))

    for label in ece_labels:
        ece_curves[ece_labels.index(label)] = mean_intensity_curves[label]

    benign_curves = np.delete(mean_intensity_curves, ece_labels, axis=0)
    return ece_curves, benign_curves


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
