import os
from pathlib import Path

import pandas as pd

from meso_regions.mesoECE.experiment import Experiment
from meso_regions.run.config import define_config_slic, define_config_ece, \
    define_config_superspels, define_config_disf, define_config_full_ece

path_classes = Path("/data_lids/home/taylla/PycharmProjects/meso/data"
                    "/classes_subclasses_nodular.csv")

path_images = Path("/data_lids/home/taylla/PycharmProjects/meso/data/resample"
                   "/images_orig_reg")
path_masks_dilated = Path("/data_lids/home/taylla/PycharmProjects/meso/output/"
                          "/HyperparameterTuning/Dilation")

path_output = Path("/data_lids/home/taylla/PycharmProjects/meso/output"
                   "/HyperparameterTuning/No_P_masks_meso_bape")
path_methods_train = Path("/data_lids/home/taylla/PycharmProjects/meso/output"
                          "/HyperparameterTuning/No_P_masks/")
path_splits = Path("/data_lids/home/taylla/PycharmProjects/meso/data/splits")
# ids_test = [7,12]
ids_test = list(sorted(
    (pd.read_csv("/data_lids/home/taylla/PycharmProjects/meso/data/splits"
                 "/test_set_bape_meso_4.csv")["ID"]).to_list()))
threads = min(os.cpu_count(), len(ids_test))


# Function to determine the mask path based on parameters
def get_mask_path(mask_name):
    return path_masks_dilated / mask_name / 'Dilate'


# Function to configure experiment settings based on method and parameters
def get_configuration(method, parts):
    if 'otsu' in parts:
        idx_otsu = parts.index('otsu')
        config_details = parts[idx_otsu + 1:]
    else:
        config_details = parts[5:]
    config = {}
    if method == 'SLIC_meso_bape':
        n_segment, compactness, p_seeds_final = config_details
        config['config_slic'] = define_config_slic(n_segments=int(n_segment),
                                                   compactness=float(
                                                       compactness),
                                                   p_seeds_final=float(
                                                       p_seeds_final))
        config['config_superspels'] = define_config_superspels()
        config['config_ece'] = define_config_ece()
    # elif method == 'DISF':
    #     n_final, p_seeds_final = config_details
    #     config['config_disf'] = define_config_disf(
    #         n_init=int(n_final) * 10,
    #         n_final=int(n_final),
    #         p_seeds_init=float(p_seeds_final) * 10,
    #         p_seeds_final=float(p_seeds_final))
    #     config['config_superspels'] = define_config_superspels()
    #     config['config_ece'] = define_config_ece()
    # elif method == 'fullece':
    #     f = config_details[0]
    #     config['config_full_ece'] = define_config_full_ece(ref_t=270,
    #                                                        filter_size=int(f),
    #                                                        with_mask=True)
    return config


metrics_experiment = {}
# Iterate over each method directory in the train metrics directory
for method_dir in path_methods_train.iterdir():

    if method_dir.name == 'SLIC_meso_bape':
        print(method_dir)
        df_metrics = pd.read_csv(method_dir / 'train' / "metrics_all_exp.csv",
                                 index_col=0)

        # Dictionary to hold the best metrics
        best_metrics = {}
        for metric in ['AUC', 'acc', 'f1_score', 'sensitivity', 'specificity']:
            best_col = df_metrics.loc[metric].idxmax()
            best_metrics[metric] = {'Value': df_metrics.at[metric, best_col],
                                    'Parameters': best_col}

        # Save the best metrics to CSV
        df_best_metrics = pd.DataFrame(best_metrics)
        df_best_metrics.to_csv(
            method_dir / 'train' / f"best_metrics_{method_dir.name}.csv")

        # Prepare for experiment
        for metric, metric_info in best_metrics.items():
            params = metric_info['Parameters']
            parts = params.split('_')
            if 'otsu' in parts:
                idx_otsu = parts.index('otsu')
                mask_name = '_'.join(parts[:idx_otsu])
            else:
                mask_name = '_'.join(parts[:5])

            # print(mask_name)
            path_masks = get_mask_path(mask_name)
            print(path_masks)
            config = get_configuration(method_dir.name, parts)
            experiment = Experiment(
                path_masks=path_masks,
                ids=ids_test,
                path_classes=path_classes,
                path_images=path_images,
                path_experiments=path_output,
                experiment_name=f"{method_dir.name}/{metric}",
                config=[c for c in config.values()],
                threads=threads
            )
            metrics_result = experiment.execute_pipeline()
            print(metrics_result)
            metrics_experiment[f"{method_dir.name}_{metric}"] = metrics_result

# df = pd.DataFrame(metrics_experiment)
# df.to_csv(path_output / "metrics_experiment.csv")
