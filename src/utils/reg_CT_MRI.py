import ants
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import os
from pathlib import Path

from tqdm import tqdm

from src.mesoECE.data_structure.mrimage import MRImage
import skimage

path_ct_images = Path('/app/data/mesexp/data/ct_scans/ct_imgs')
path_ct_mask_fluid = Path('/app/data/mesexp/data/ct_scans/ct_mask')
path_ct_mask_lung = Path('/app/data/mesexp/data/ct_scans/ct_lung_mask')

#path_mri_images = Path('/app/data/src/data/images')
path_mri_images = Path('/app/data/mesexp/data/images_orig')
path_out = Path('/app/data/mesexp/data/REG-CT-MRI/pico_CT_pleura_N4_6')
path_mri_mask_lung = path_out / 'mri_mask_lung/new_70'

patients_ct = ['0070.nii.gz']

for patient_ct in tqdm(patients_ct):

    # Get patient id
    id = int(patient_ct.split('.')[0])
    print(f'\nPATIENT: {id}')
    # Loading MRI and CT scans and CT mask
    if id == 287:
        mri_patient_filename = MRImage.resolve_name(id, 540)
    else:
        mri_patient_filename = MRImage.resolve_name(id, 270)
    mri_img = nib.load(path_mri_images / mri_patient_filename)

    ct_img = nib.load(path_ct_images / patient_ct)
    ct_mask = nib.load(path_ct_mask_fluid / patient_ct)
    ct_lung_mask = nib.load(path_ct_mask_lung / patient_ct)

    # Get numpy array
    mri_data = mri_img.get_fdata()
    ct_data = ct_img.get_fdata()
    ct_mask_data = ct_mask.get_fdata()
    ct_lung_mask_data = ct_lung_mask.get_fdata()

    # Get the dimensions of the MRI and CT scans
    mri_shape = mri_data.shape
    ct_shape = ct_data.shape
    ct_mask_shape = ct_mask_data.shape
    ct_lung_mask_shape = ct_lung_mask_data.shape

    print(f'MRI Scan Shape: {mri_shape}')
    print(f'CT Scan Shape: {ct_shape}')
    if(ct_data.shape[2]>460):
        ct_data = ct_data[:, :, -460:]
        ct_mask_data = ct_mask_data[:, :, -460:]
        ct_lung_mask_data = ct_lung_mask_data[:, :, -460:]


    print('Adding the mask to the CT scan...')
    ct_img_masked = ct_data.copy()  # Create a copy of the original data
    min_in_mask = ct_img_masked[ct_mask_data > 0].min()
    min_in_mask = ct_img_masked[ct_mask_data > 0].min()
    ct_img_masked[ct_mask_data > 0] = (((ct_img_masked[ct_mask_data > 0] - min_in_mask) / np.max(
        ct_data)) ** 0.6) * np.max(ct_data) + min_in_mask

    print('Normalizing images...')
    ct_img_prepross = skimage.exposure.rescale_intensity(ct_img_masked, in_range='image', out_range=(0, 1))
    mri_img_prepross = skimage.exposure.rescale_intensity(mri_data, in_range='image', out_range=(0, 1))

    ct_img_ant = ants.utils.convert_nibabel.from_nibabel(
        nib.Nifti1Image(ct_img_prepross, affine=ct_img.affine, header=ct_img.header))

    mri_img_ant = ants.utils.convert_nibabel.from_nibabel(
        nib.Nifti1Image(mri_img_prepross, affine=mri_img.affine, header=mri_img.header))

    print('Resampling CT scan and mask...')

    mri_spacing = mri_img.header.get_zooms()
    ct_img_ant_resampled = ants.resample_image(ct_img_ant, mri_spacing, use_voxels=False, interp_type=4)

    ct_mask_ants = ants.utils.convert_nibabel.from_nibabel(
        nib.Nifti1Image(ct_mask_data, affine=ct_mask.affine))
    ct_mask_ants_resampled = ants.resample_image_to_target(ct_mask_ants, ct_img_ant_resampled, verbose=True,
                                                           interp_type='genericLabel')

    ct_lung_mask_ants = ants.utils.convert_nibabel.from_nibabel(
        nib.Nifti1Image(ct_lung_mask_data[:,::-1,:], affine=ct_mask.affine))
    ct_lung_mask_ants_resampled = ants.resample_image_to_target(ct_lung_mask_ants, ct_img_ant_resampled,
                                                           interp_type='genericLabel')

    print('Saving Resampled CT scan and mask...')
    if not os.path.exists(path_out / str(id)):
        os.makedirs(path_out / str(id))
    nib.save(ants.utils.convert_nibabel.to_nibabel(ct_img_ant_resampled),
             str(path_out / str(id) / 'ct_img_resampled.nii.gz'))
    nib.save(ants.utils.convert_nibabel.to_nibabel(ct_mask_ants_resampled),
             str(path_out / str(id) / 'ct_mask_resampled.nii.gz'))
    nib.save(ants.utils.convert_nibabel.to_nibabel(ct_lung_mask_ants_resampled),
             str(path_out / str(id) / 'ct_lung_mask_resampled.nii.gz'))

    # registration
    print('Registration of CT Scan and MRI Scan...')
    reg = ants.registration(fixed=mri_img_ant, moving=ct_img_ant_resampled, type_of_transform='SyNCC') #restrict_transformation= (0,0,1))

    warped_fix_img = ants.utils.convert_nibabel.to_nibabel(reg["warpedfixout"])
    warped_mov_img = ants.utils.convert_nibabel.to_nibabel(reg["warpedmovout"])
    transform_fwd = ants.read_transform(reg["fwdtransforms"][1])
    transform_inv = ants.read_transform(reg["invtransforms"][0])


    print('Saving registration output imgs and transforms..')
    if not os.path.exists(path_out / str(id) / 'reg'):
        os.makedirs(path_out / str(id) / 'reg')

    nib.save(warped_fix_img, str(path_out / str(id) / 'reg' / 'warped_fix_img.nii.gz'))
    nib.save(warped_mov_img, str(path_out / str(id) / 'reg' / 'warped_mov_img.nii.gz'))

    # saving transforms (forward and inverted) from the registration
    transform_fwd_filename = str(id) + "_fwd.mat"
    transform_inv_filename = str(id) + "_inv.mat"

    ants.write_transform(transform_fwd, str(path_out / str(id) / 'reg' / transform_fwd_filename))
    ants.write_transform(transform_inv, str(path_out / str(id) / 'reg' / transform_inv_filename))

    print('Applying forward transform to obtain MRI Mask...')
    #mri_mask = ants.apply_transforms(fixed=mri_img_ant, moving=ct_mask_ants_resampled, interpolator="genericLabel",
    #                                 transformlist=reg['fwdtransforms'])
    mri_lung_mask = ants.apply_transforms(fixed=mri_img_ant, moving=ct_lung_mask_ants_resampled, interpolator="genericLabel",
                                     transformlist=reg['fwdtransforms'])

    if not os.path.exists(path_mri_mask_lung):
        os.makedirs(path_mri_mask_lung)
    print('Saving MRI mask...')

    if id == 287:
        mri_mask_filename = MRImage.resolve_name(id, 540)
    else:
        mri_mask_filename = MRImage.resolve_name(id, 270)
    #nib.save(mri_mask, str(path_mri_mask / mri_mask_filename))
    nib.save(mri_lung_mask, str(path_mri_mask_lung / mri_mask_filename))