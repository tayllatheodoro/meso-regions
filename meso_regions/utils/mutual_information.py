import os
from pathlib import Path
import nibabel as nib
import numpy as np
from matplotlib import pyplot as plt
from skimage import io, filters
from skimage.morphology import remove_small_objects
import pandas as pd

path_images = Path(
    '/data_lids/home/taylla/PycharmProjects/meso/data/orig/images')
path_classes = Path("/data_lids/home/taylla/PycharmProjects/meso/data"
                    "/classes_subclasses_nodular.csv")

path_images_reg = Path(
    '/data_lids/home/taylla/PycharmProjects/meso/output/ants_reg/ANTs/images')

path_output = Path(
    '/data_lids/home/taylla/PycharmProjects/meso/output/qualitative_analysis/registration')
os.makedirs(path_output, exist_ok=True)
ids = list(sorted((pd.read_csv(path_classes)["ID"]).to_list()))

from sklearn.metrics import mutual_info_score


def calculate_mutual_information(image1, image2, n_bins=256):
    """
    Calculate the mutual information between two images using histogram binning and save the result to a CSV file.

    Parameters:
    image1 (numpy.ndarray): First image (2D or 3D).
    image2 (numpy.ndarray): Second image (2D or 3D).
    n_bins (int): Number of bins for histogram.
    """
    # Flatten the images to 1D arrays
    img1_flat = image1.ravel()
    img2_flat = image2.ravel()

    # Histogram binning
    max_value = max(image1.max(), image2.max())
    min_value = min(image1.min(), image2.min())
    bins = np.linspace(min_value, max_value, n_bins)

    img1_binned = np.digitize(img1_flat, bins)
    img2_binned = np.digitize(img2_flat, bins)

    #Calculate mutual information
    mi = mutual_info_score(img1_binned, img2_binned)

    return mi


filename_images = os.listdir(path_images_reg)
mutual_info_dir_reg = {}
mutual_info_dir = {}

mutual_info_dir_diff = {}
mi_list_orig = []
mi_list_reg = []
mi_list_diff = []

for id in ids: # patients

    list_filename_images = [f for f in filename_images if
                            f.startswith(f'{id:05d}')]

    list_filename_images.remove(f'{id:05d}_270.nii.gz')
    print(list_filename_images)

    img_270_orig = nib.load(path_images / f'{id:05d}_270.nii.gz')
    img_270_data_orig = img_270_orig.get_fdata()

    img_270_reg = nib.load(path_images_reg / f'{id:05d}_270.nii.gz')
    img_270_data_reg = img_270_reg.get_fdata()

    mutual_info_orig_list = []
    mutual_info_reg_list = []
    time_points = []
    for filename_image in list_filename_images:
        t = int(filename_image.split('_')[1].split('.')[0])

        img_orig = nib.load(path_images / filename_image)
        img_data_orig = img_orig.get_fdata()
        mi_score_orig = calculate_mutual_information(img_270_data_orig, img_data_orig)
        mutual_info_orig_list.append(mi_score_orig)
        time_points.append(t)

        img_reg = nib.load(path_images_reg / filename_image)
        img_data_reg = img_reg.get_fdata()
        mi_score_reg = calculate_mutual_information(img_270_data_reg, img_data_reg)
        mutual_info_reg_list.append(mi_score_reg)

        mutual_info_dir_reg[f'{id}_{t}'] = mi_score_reg
        mutual_info_dir[f'{id}_{t}'] = mi_score_orig
        mutual_info_dir_diff[f'{id}_{t}'] = mi_score_reg- mi_score_orig

    # mean_mi_orig = np.mean(mutual_info_orig_list)
    # mean_mi_reg = np.mean(mutual_info_reg_list)
    #
    # mutual_info_dir_reg[f'{id}'] = mean_mi_reg
    # mutual_info_dir[f'{id}'] = mean_mi_orig
    # mutual_info_dir_diff[f'{id}'] = mean_mi_reg - mean_mi_orig

    # Sort time points and mutual information lists together
    sorted_tuples = sorted(zip(time_points, mutual_info_orig_list))
    time_points, mutual_info_orig_list = zip(
        *sorted_tuples)  # Unpacking the sorted tuples back into separate lists

    sorted_tuples_reg = sorted(zip(time_points, mutual_info_reg_list))
    time_points, mutual_info_reg_list = zip(*sorted_tuples_reg)

    mi_list_orig.append((id, time_points, mutual_info_orig_list))
    mi_list_reg.append((id, time_points, mutual_info_reg_list))

    # mi_list_orig.append((id, time_points, mutual_info_orig_list))
    #
    # mi_list_reg.append((id, time_points, mutual_info_reg_list))
    # mi_list_mean_diff.append(mean_mi_reg - mean_mi_orig)

# sort the list by time points


#plot all list orig in the same plot id is the patient,


for id_info in mi_list_orig:
    patient_id, time_points, mis = id_info
    plt.plot(time_points, mis, label=f'Patient {patient_id}')

plt.xlabel('Time Points')
plt.ylabel('Mutual Information')
plt.title('Mutual Information Over Time')
# plt.legend()
plt.show()

#plt.figure(figsize=(10, 6))
#mean = np.mean(non_zero_curve, axis=0)
# std = np.std(non_zero_curve, axis=0)
#
# plt.plot(time_points, mean, label="Mean Curve", color="blue")
# plt.fill_between(time_points, mean - std, mean + std, color="blue",
#                  alpha=0.2)
# plt.axvline(x=270, color='red')
#
# plt.xlabel("Time Points")
# plt.ylabel("")
# plt.title(f"Mean_{title}")
# plt.grid(True)


# plt.xlabel('Time points')
# plt.ylabel('Mutual Information')
# plt.legend()
# plt.show()


#
# plt.stairs(mi_list_mean_orig, label='Original', color='blue')
# plt.stairs(mi_list_mean_reg, label='Registered', color='red')
# # plt.plot(mi_list_mean_diff, label='Difference')
# plt.xlabel('Patients')
# plt.ylabel('Mutual Information')
# plt.legend()
# plt.savefig(path_output / 'mutual_information.png')
#


# #save the mutual information to a csv file
df = pd.DataFrame.from_dict(mutual_info_dir_reg, orient='index')
df.to_csv(path_output / 'mutual_information_reg.csv')

df = pd.DataFrame.from_dict(mutual_info_dir, orient='index')
df.to_csv(path_output / 'mutual_information_orig.csv')

df = pd.DataFrame.from_dict(mutual_info_dir_diff, orient='index')
df.to_csv(path_output / 'mutual_information_diff.csv')
#

