# This file contains functions to define configuration dictionaries for each module.

## module: img_preprocess
def define_ant_inv_mask(path_transforms,ref_t=270):
    """Define configuration for ANTsInvMask method.

    Args:
        -ref_t              Reference time point.
        -path_transforms    Path to the transforms dir for the inverse
                            transformation.
    Returns:
        -config_ant_inv_mask    Configuration dictionary for ANTsInvMask method.
        """
    config_ant_inv_mask = {
        "module": "img_preprocess",
        "method": "ANTsInvMask",
        "args": {"ref_t": ref_t, "path_transforms": path_transforms}
    }
    return config_ant_inv_mask


def define_ants_reg(ref_t: int = 270, domain: str = None, **config):
    """Define configuration for ANTsReg method.

    Args:
        -ref_t          Reference time point.
        -domain         Domain of the analysis of ANTsReg [REG, ORIG].
        -config         Configuration dictionary for ANTs registration.
    Returns:
        -config_ants_reg    Configuration dictionary for ANTsReg method.
    """
    config_ants_reg = {
        "module": "img_preprocess",
        "method": "ANTs",
        "args": {"ref_t": ref_t, "domain": domain, "config": config}
    }
    return config_ants_reg


def define_bias_correction():
    """Define configuration for BiasCorrection method with N4BiasCorrection.

    Returns:
        -config_bias_correction    Configuration dictionary for BiasCorrection
                                    method.
    """
    config_bias_correction = {
        "module": "img_preprocess",
        "method": "BiasCorrection",
        "args": {}
    }
    return config_bias_correction


def define_resample(ref_t: int = 270, img_spacing: float = 1.78, img_dim=None):
    """Define configuration for Resample method.

    Args:
        -ref_t          Reference time point.
        -img_spacing    Target voxel size. If 0, the image will be resampled to
                        the image dim size. Default: 1.78.
        -img_dim        Target image dimensions.
    Returns:
        -config_resample    Configuration dictionary for Resample method.
        """
    config_resample = {
        "module": "img_preprocess",
        "method": "Resample",
        "args": {"ref_t": ref_t, "img_spacing": img_spacing,
                 "img_dim": img_dim}
    }
    return config_resample


## module: mask_preprocess
def define_config_dilate(dilation_radius, p_center_distance,ref_t =270, otsu = False,
                         mask_to_dilate = 'fluid'):
    """Define configuration for Dilation method.

    Args:
        -ref_t              Reference time point.
        -dilation_radius    Radius of the dilation.
        -p_center_distance  Percentage of the distance to the center of the
                            image to start dilatation.
        -otsu               Apply Otsu thresholding to the output mask.
                            Default:False.
        -mask_to_dilate     Mask to dilate.
    Returns:
        -config_dilation    Configuration dictionary for Dilation method.
    """
    config_dilation = {
        "module": "mask_preprocess",
        "method": "Dilate",
        "args": {"ref_t": ref_t, "dilation_radius": dilation_radius,
                 "p_center_distance": p_center_distance, "otsu": otsu,
                 "mask_to_dilate": mask_to_dilate}
    }
    return config_dilation


def define_config_add_lung(ref_t=270):
    """Define configuration for AddLung method.

    Add lung to the fluid mask before dilating it.
    Args:
        -ref_t          Reference time point.
    Returns:
        -config_add_lung    Configuration dictionary for AddLung method.
    """
    config_add_lung = {
        "module": "mask_preprocess",
        "method": "AddLung",
        "args": {"ref_t": ref_t}
    }
    return config_add_lung


def define_config_sub_lung(ref_t=270):
    """Define configuration for SubLung method.

    Sub lung from the fluid mask after  dilating it.
    Args:
        -ref_t          Reference time point.
    Returns:
        -config_sub_lung    Configuration dictionary for SubLung method.
    """
    config_sub_lung = {
        "module": "mask_preprocess",
        "method": "SubLung",
        "args": {"ref_t": ref_t}
    }
    return config_sub_lung


## module: supervoxels
def define_config_slic(n_segments, compactness, p_seeds_final,ref_t=270):
    """Define configuration for SLIC method.

    Args:
        -ref_t          Reference time point.
        -n_segments     Number of segments. If 0, it will be calculated as a
                        percentage of the volume.
        -compactness    Compactness parameter.
        -p_seeds_final  Percentage of seeds.
    Returns:
        -config_slic    Configuration dictionary for SLIC method.
    """
    config_slic = {
        "module": "supervoxel",
        "method": "SLIC",
        "args": {"ref_t": ref_t, "n_segments": n_segments,
                 "compactness": compactness, "p_seeds_final": p_seeds_final}
    }
    return config_slic


def define_config_hslic(n_segments, n_segments_h, compactness, p_seeds_final,
                         ref_t=270):
    """Define configuration for SLIC_H method.

    Args:
        -ref_t          Reference time point.
        -n_segments     Number of segments.
        -n_segments_h   Number of segments for the H-SLIC.
        -compactness    Compactness parameter.
        -p_seeds_final  Percentage of seeds.
    Returns:
        -config_slic_h    Configuration dictionary for SLIC_H method.
    """
    config_hslic = {
        "module": "supervoxel",
        "method": "HSLIC",
        "args": {"ref_t": ref_t, "n_segments": n_segments,
                 "n_segments_h": n_segments_h, "compactness": compactness,
                 "p_seeds_final": p_seeds_final}
    }
    return config_hslic


def define_config_disf(n_init, n_final, p_seeds_init, p_seeds_final,ref_t =270,
                       ift_path: str = '/data_lids/home/taylla/ift'):
    """Define configuration for DISF method.

    Args:
        -ref_t          Reference time point.
        -n_init         Number of seeds at initial time point. If 0, it will be
                        calculated as a percentage of the volume*.
        -n_final        Number of seeds at final time point. If 0, it will be
                        calculated as a percentage of the volume*.
        -p_seeds_init   Percentage of seeds at initial time point.
        -p_seeds_final  Percentage of seeds at final time point.
        *if n_init and n_final are both 0, the percentage of seeds will be
        calculated for both time points.
    Returns:
        -config_disf    Configuration dictionary for DISF method.
        """
    config_disf = {
        "module": "supervoxel",
        "method": "DISF",
        "args": {"ref_t": ref_t, "n_init": n_init, "n_final": n_final,
                 "p_seeds_init": p_seeds_init, "p_seeds_final": p_seeds_final,
                 'ift_path': ift_path}
    }
    return config_disf


## module: superpels
def define_config_superspels(ref_t=270):
    """Define configuration for Superspel method.

    Args:
        -ref_t          Reference time point.
    Returns:
        -config_superspels  Configuration dictionary for Superspel method.
        """
    config_superspels = {
        "module": "superspels",
        "method": "Superspel",
        "args": {"ref_t": ref_t}
    }
    return config_superspels


## module: classifier

def define_config_ece(ref_t=270):
    """Define configuration for ECE method.

    Args:
        -ref_t          Reference time point.
    Returns:
        -config_ece     Configuration dictionary for ECE method.
        """
    config_ece = {
        "module": "classifier",
        "method": "ECE",
        "args": {"ref_t": ref_t}
    }
    return config_ece


## module : supervoxel_with_ece

def define_config_full_ece(filter_size, with_mask,ref_t=270):
    """Define configuration for FullECE method.

    Args:
        -ref_t          Reference time point.

    Returns:
        -config_full_ece    Configuration dictionary for FullECE method.
    """
    config_full_ece = {
        "module": "supervoxel_with_ece",
        "method": "FullECE",
        "args": {"ref_t": ref_t, "filter_size": filter_size,
                 "with_mask": with_mask}
    }
    return config_full_ece


def define_config_div_with_ece(n_segments: int,
                               compactness: float,ref_t =270, p_size: int =1000,
                               method: str = 'SLIC',
                               predict_only_small: bool = False):
    """Define configuration for DivideWithECE method.

    Args:
        -ref_t                  Reference time point.
        -n_segments             Number of segments.
        -compactness            Compactness parameter for SLIC.
        -p_size                 Quantity of Pixels for regions.
        -method                 Method to use for the division
                                    Options: [DISF, SLIC, SICLE].
                                    Default: SLIC.
        -predict_only_small     Predict only small patches.
        -decrease_n_segments    Decrease the number of segments.
    Returns:
        -config_div_with_ece    Configuration dictionary for DividingECE method.
    """
    config_div_with_ece = {
        "module": "supervoxel_with_ece",
        "method": "DivideWithECE",
        "args": {"ref_t": ref_t, "n_segments": n_segments,
                 "compactness": compactness, "p_size": p_size,
                 "method": method,
                 "predict_only_small": predict_only_small}
    }
    return config_div_with_ece
