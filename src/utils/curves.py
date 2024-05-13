from pathlib import Path
import os

import numpy as np
from scipy.interpolate import interp1d

import pandas as pd
from tqdm import tqdm

path_classes = Path(
    "/data_lids/home/taylla/PycharmProjects/meso/data/classes_subclasses_nodular.csv")
path_slic = Path(
    "/data_lids/home/taylla/PycharmProjects/meso/output/HyperparameterTuning/No_P_masks/SLIC/test")
path_ids_test = Path(
    "/data_lids/home/taylla/PycharmProjects/meso/data/splits/test_set_classes_4.csv")
path_disf = Path(
    "/data_lids/home/taylla/PycharmProjects/meso/output/HyperparameterTuning/No_P_masks/DISF/test")
path_out = Path("/data_lids/home/taylla/PycharmProjects/meso/output/Attention_test")



path_method = path_disf
method = 2
standard_time_points = [0, 40, 80, 180, 270, 540, 810]
os.makedirs(path_out, exist_ok=True)
# list dirs not csv
list_masks = [x for x in os.listdir(path_method) if os.path.isdir(path_method / x)]
ids_train = list(sorted(
    (pd.read_csv(path_ids_test)["ID"]).to_list()))
superspels = {}

classes_df = pd.read_csv(path_ids_test)
classes_df.set_index('ID', inplace=True)
for id in tqdm(ids_train):
    super_count = 0
    for m in list_masks:
        print(m)

        list_superspels = os.listdir(path_method / m)
        for i, s in enumerate(list_superspels):

            diagnosis = classes_df.loc[id, 'CLASS']

            mean_df = pd.read_csv(
                path_method / m / s / 'Superspel' / 'curves_df' / f'mean_intensity_curves_{id}.csv')
            mean_df = mean_df.drop(mean_df.index[0])
            current_time_points = mean_df.columns.tolist()
            current_time_points = [int(s) for s in current_time_points]

            std_df = pd.read_csv(
                path_method / m / s / 'Superspel' / 'curves_df' / f'std_intensity_curves_{id}.csv')
            std_df = std_df.drop(std_df.index[0])
            mean_curves = mean_df.to_numpy()
            std_curves = std_df.to_numpy()

            new_mean_curves = np.zeros((mean_curves.shape[0], len(standard_time_points)))
            new_std_curves = np.zeros((std_curves.shape[0], len(standard_time_points)))

            # if not all time points are present, interpolate
            for curve in range(mean_curves.shape[0]):
                f_interp = interp1d(current_time_points, mean_curves[curve, :],
                                    fill_value=mean_curves[curve][current_time_points.index(current_time_points[-1])],
                                    bounds_error=False)

                for j, time in enumerate(standard_time_points):

                    if time not in current_time_points:
                        new_mean_curves[curve, j] = f_interp(time)
                    else:
                        new_mean_curves[curve, j] = mean_curves[curve, current_time_points.index(time)]
            for curve in range(std_curves.shape[0]):
                f_interp = interp1d(current_time_points, std_curves[curve, :],
                                    fill_value=std_curves[curve][-1],
                                    bounds_error=False)
                for j, time in enumerate(standard_time_points):
                    if time not in current_time_points:
                        new_std_curves[curve, j] = f_interp(time)
                    else:
                        new_std_curves[curve, j] = std_curves[curve, current_time_points.index(time)]
            #concat mean and std
            mean_std = np.concatenate((new_mean_curves, new_std_curves), axis=1)
            # to csv
            pd.DataFrame(mean_std).to_csv(path_out / f'{diagnosis+1:03d}_{method:003d}_{super_count:03d}_{id:05d}.csv', index=False, header=False)
            super_count += 1













