from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import skimage
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from meso_regions.mesoECE.data_structure import Patient
from meso_regions.mesoECE.methods.utils import setup_directories


def plot_curves(curve: np.ndarray, mask: np.ndarray, time_points: list[int],
                filename: Union[str, Path] = None, title: str = 'Curves',
                mean_plot: bool = False) -> None:
    plt.figure(figsize=(10, 6))

    labels = np.unique(mask[mask > 0])
    non_zero_curve = np.zeros((labels.shape[0], curve.shape[1]))
    for i, label in enumerate(labels):
        if np.sum(curve[int(label)]) > 0:
            plt.plot(time_points, curve[int(label)],
                     label=f'Label {int(label)}')
            non_zero_curve[i] = curve[int(label)]

    plt.xlabel('Time Points')
    plt.ylabel('Mean Intensity')
    # Add a red vertical line at x=3
    plt.axvline(x=270, color='red')
    plt.title(title)
    plt.grid(True)

    if filename:
        plt.savefig(f'{filename}.png')  # Save the plot as a PNG file

    else:
        plt.show()
    plt.close('all')

    if mean_plot:
        plt.figure(figsize=(10, 6))
        mean = np.mean(non_zero_curve, axis=0)
        std = np.std(non_zero_curve, axis=0)

        plt.plot(time_points, mean, label="Mean Curve", color="blue")
        plt.fill_between(time_points, mean - std, mean + std, color="blue",
                         alpha=0.2)
        plt.axvline(x=270, color='red')

        plt.xlabel("Time Points")
        plt.ylabel("")
        plt.title(f"Mean_{title}")
        plt.grid(True)
        if filename is not None:
            plt.savefig(f"{filename}_mean.png")
        else:
            plt.show()
        plt.close('all')


def save_and_plot_curves(path: Path, patient: Patient, curves: np.ndarray,
                         curve_name: str, mask: np.ndarray) -> None:
    setup_directories(path, ['plots', 'curves_df', f'plots/{curve_name}'])
    curve_path = path / 'curves_df'
    plot_path = path / 'plots' / curve_name

    # Save curves and interpolation to CSV
    save_curves_and_interp_to_csv(
        patient=patient,
        curves=curves,
        path=curve_path,
        curve_name=curve_name
    )

    # Plot curves
    plot_curves(
        curve=curves,
        time_points=patient.time_points,
        mask=mask,
        filename=plot_path / f'{patient.id}',
        mean_plot=True,
        title=f'All Curves - {curve_name.capitalize()}'
    )


def save_curves_to_csv(curves: np.ndarray, time_points: list[int],
                       filename: str = None) -> None:
    column_names_str = list(map(str, time_points))
    df = pd.DataFrame(curves, columns=column_names_str)
    df.to_csv(filename, index=False)


def interp_curve_missing_time_points(curves: np.ndarray,
                                     time_points: list[int]) -> tuple[
    np.ndarray, list[int]]:
    standard_time_points = [0, 40, 80, 180, 270, 540, 810]

    new_curves = np.zeros((curves.shape[0], len(standard_time_points)))

    # if not all time points are present, interpolate
    for curve in range(curves.shape[0]):
        f_interp = interp1d(time_points, curves[curve, :],
                            fill_value=curves[curve][
                                time_points.index(
                                    time_points[-1])],
                            bounds_error=False)

        for j, time in enumerate(standard_time_points):

            if time not in time_points:
                new_curves[curve, j] = f_interp(time)
            else:
                new_curves[curve, j] = curves[
                    curve, time_points.index(time)]
    return new_curves, standard_time_points


def save_curves_and_interp_to_csv(path: Path, patient: Patient,
                                  curves: np.ndarray,
                                  curve_name: str) -> None:
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


def define_mean_std_intensity_curves(patient: Patient, mask: np.ndarray) -> \
tuple[np.ndarray, np.ndarray]:
    mean_intensity_curves = np.zeros(
        (int(mask.max()) + 1,
         patient.time_points.__len__()))
    std_intensity_curves = np.zeros_like(
        mean_intensity_curves)

    rps = skimage.measure.regionprops(mask)
    for rp in rps:
        slice_bbox = tuple(
            [slice(dim_start, dim_finish) for dim_start, dim_finish in
             zip(rp.bbox[:3], rp.bbox[3:])])
        lbl_in_bbox = rp.image

        for t in patient.time_points:
            img = patient.get_image(t).data
            img_in_bbox = img[slice_bbox] - patient.background_otsu(t)
            mean_intensity_curves[rp.label, patient.time_points.index(t)] = \
                img_in_bbox[lbl_in_bbox > 0].mean()
            std_intensity_curves[rp.label, patient.time_points.index(t)] = \
                img_in_bbox[lbl_in_bbox > 0].std()

    return mean_intensity_curves, std_intensity_curves


def calculate_curves(patient: Patient,
                     ref_t=270,
                     mask_name='supervoxels') -> tuple[np.ndarray, np.ndarray]:
    mask = patient.get_image(ref_t).masks[mask_name].data.astype(
        np.int32)
    return define_mean_std_intensity_curves(patient=patient, mask=mask)
