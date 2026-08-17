import nibabel as nib
import os
import numpy as np
from pathlib import Path
import skimage
from matplotlib import pyplot as plt
import ants

from collections import Counter

path_mask = Path('/app/data/mesexp/data/REG-CT-MRI/pico_CT_pleura_N4_6/mri_mask_lung/')
path_imgs = Path('/app/data/mesexp/data/images_orig')
path_fluid_mask = Path('/app/data/mesexp/data/fluid_mask')
path_out = Path('/app/data/mesexp/data/REG-CT-MRI/pico_CT_pleura_N4_6/mri_mask_lung_corrected/otsu_025')
os.makedirs(path_out, exist_ok=True)
os.listdir(path_mask)

spacing = []
for i in os.listdir(path_imgs):

    input_image = ants.image_read(str(path_imgs / i))

    # Get the spacing (voxel sizes) of the image
    spacing.append(input_image.spacing)


counted_values = Counter(spacing)

for value, count in counted_values.items():
    print(f"{value} is repeated {count} times.")