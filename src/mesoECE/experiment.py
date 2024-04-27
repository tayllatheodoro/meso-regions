import shutil
from pathlib import Path

import json
import os
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
from scipy.stats import stats
from sklearn import metrics

import numpy as np
import pandas as pd
import src.mesoECE.methods.mask_preprocess as mask_preprocess_module
import src.mesoECE.methods.img_preprocess as img_preprocess_module
import src.mesoECE.methods.supervoxel as supervoxel_module
import src.mesoECE.methods.superspels as superspels_module
import \
    src.mesoECE.methods.supervoxel_with_classifier as supervoxel_with_ece_module

import src.mesoECE.methods.classifier as classifier_module
from src.mesoECE.data_structure import MesoDataset


class Experiment:
    def __init__(self,
                 experiment_name: str,
                 path_images: Path,
                 path_masks: Path,
                 path_experiments: Path,
                 path_classes: Path,
                 ids: List[int],
                 config: List[dict],
                 threads: int = 0):
        self.classifier = None
        self.path_prev_exp = None
        self.results = None
        self.config = config
        self.path_exp = path_experiments / experiment_name
        os.makedirs(self.path_exp, exist_ok=True)

        self.pipeline_methods = []

        self.instantiate_experiment()

        masks_list = os.listdir(path_masks)
        # masks_dir = None
        # if len(masks_list) > 1:
        #     for mask in masks_list:
        #         masks_dir = {mask: path_masks / mask}
        # elif masks_list[0] == 'fluid':
        #     masks_dir = {'fluid': path_masks/'fluid'}
        # # else:
        masks_dir = {'pleural_region': path_masks}

        self.image_dataset = MesoDataset(path_images=path_images,
                                         path_classes=path_classes,
                                         path_masks=masks_dir, ids=ids,
                                         threads=threads)

    def parse_experiment_partial_files(self, config: dict) -> Path:
        """
        Return the path from the experiment containing the desired configuration
        or create a new path which will be the output for the operations under
        your  configuration.
        """
        path_part_exp = self.path_exp / config["method"]

        os.makedirs(path_part_exp, exist_ok=True)

        # partial_exp_dirs = sorted(os.listdir(path_part_exp))
        #
        # if len(partial_exp_dirs) > 0:
        #     # try to read the experiment config and data
        #     for part_exp in partial_exp_dirs:
        #         with open(path_part_exp / part_exp / 'config.json') as j_file:
        #             current_config = json.load(j_file)
        #             if current_config == config:
        #                 return path_part_exp / part_exp
        #     id_exp = int(partial_exp_dirs[-1]) + 1
        # else:
        #     id_exp = 1
        #
        # # create a new experiment dir
        # new_dir = f"{id_exp:05d}"
        # os.makedirs(path_part_exp / new_dir, exist_ok=True)
        # json_out_file_path = path_part_exp / new_dir / 'config.json'
        json_out_file_path = path_part_exp / 'config.json'

        with open(json_out_file_path, "w") as j_file:
            json.dump(config, j_file)

        return path_part_exp

    def instantiate_experiment(self):
        config_list = self.config

        mapping_module = {
            "mask_preprocess": mask_preprocess_module,
            "img_preprocess": img_preprocess_module,
            "superspels": superspels_module,
            "supervoxel_with_ece": supervoxel_with_ece_module,
            "supervoxel": supervoxel_module,
            "classifier": classifier_module
        }

        self.path_prev_exp = ""
        self.pipeline_methods = []

        for config in config_list:
            method_module = config["module"]
            method_name = config["method"]
            args = config["args"]
            config["previous_experiment"] = str(self.path_prev_exp)

            module = mapping_module[method_module]
            method_class = getattr(module, method_name)

            path_part_exp = self.parse_experiment_partial_files(config)

            instance = method_class(path=path_part_exp, **args)

            self.pipeline_methods.append(instance)

            self.path_prev_exp = path_part_exp
            # save config in a file for all methods
            print("\n")
            print(config)

            with open(path_part_exp / 'config_all.json', "w") as j_file:
                json.dump(config_list, j_file)

    def execute_pipeline(self):
        self.results = {}
        for method in self.pipeline_methods:
            self.image_dataset.apply_method(method)
            if method.result() is not None:
                self.results[method.__class__.__name__] = method.result()

        #self.result_classifier()
        # self.results_seeds()
        #self.classifier_metrics()

    def result_analysis(self):

        pass

    def result_classifier(self):

        # check if ECE is in the methods
        self.classifier = ""
        if 'ECE' in self.results:
            self.classifier = "ECE"
        elif 'DivideWithECE' in self.results:
            self.classifier = "DivideWithECE"
        elif 'FullECE' in self.results:
            self.classifier = "FullECE"

        df = pd.DataFrame(self.results[self.classifier],
                          columns=['IDs', 'GRUND_TRUTH', 'ECE', 'SUBCLASS',
                                   'NODULAR', 'QT_SUPERVOXELS',
                                   'VOLUME_SUPERVOXELS'])
        path_results = self.path_prev_exp / 'diagnosis.csv'
        df.to_csv(str(path_results), index=False)

    def results_seeds(self):
        supervoxel = ""
        if 'SLIC' in self.results:
            supervoxel = "SLIC"
        elif 'DISF' in self.results:
            supervoxel = "DISF"

        df = pd.DataFrame(self.results[supervoxel],
                          columns=['IDs', 'INITIAL_SEEDS', 'FINAL_SEEDS'])
        path_results = self.path_prev_exp / 'supervoxel_n_seeds.csv'
        df.to_csv(str(path_results), index=False)

    def classifier_metrics(self):

        results_ece = self.results[self.classifier]
        metrics_exp = {}
        ids_fp = []
        ids_fn = []

        y = [result_ece[1] for result_ece in results_ece]
        y_pred = [result_ece[2] for result_ece in results_ece]

        # add IDS of false positive and false negative to lists
        for result_ece in results_ece:
            if result_ece[1] == 1 and result_ece[2] == 0:
                ids_fn.append(result_ece[0])
            if result_ece[1] == 0 and result_ece[2] == 1:
                ids_fp.append(result_ece[0])

        metrics_exp['acc'] = metrics.accuracy_score(y, y_pred)
        metrics_exp['balanced_acc'] = metrics.balanced_accuracy_score(y, y_pred)
        metrics_exp[
            'average_precision_score'] = metrics.average_precision_score(y,
                                                                         y_pred)
        metrics_exp['f1_score'] = metrics.f1_score(y, y_pred)
        metrics_exp['AUC'] = metrics.roc_auc_score(y, y_pred)
        metrics_exp['Sensitivity'] = metrics.recall_score(y, y_pred)
        metrics_exp['Specificity'] = metrics.precision_score(y, y_pred)

        tn, fp, fn, tp = metrics.confusion_matrix(y, y_pred).ravel()
        if (tp + fp) > 0:
            metrics_exp['PPV'] = tp / (tp + fp)
        elif (tn + fn) > 0:
            metrics_exp['NPV'] = tn / (tn + fn)
        metrics_exp['FP'] = fp
        metrics_exp['FN'] = fn
        metrics_exp['TN'] = tn
        metrics_exp['TP'] = tp
        metrics_exp['FP_ID'] = ids_fp
        metrics_exp['FN_ID'] = ids_fn

        # roc_display = metrics.RocCurveDisplay.from_predictions(y, y_pred)
        # roc_display.plot()
        # plt.savefig(self.path_prev_exp / 'roc_curve.png')
        # plt.show()
        # plt.close('all')

        # save confusion matrix with sklean

        # cm = metrics.confusion_matrix(y, y_pred, normalize='all')
        # cmd = metrics.ConfusionMatrixDisplay(cm,
        #                                      display_labels=['NON-MALIGNANT',
        #                                                      'MALIGNANT'])
        # cmd.plot()
        # cmd.figure_.savefig(
        #     str(self.path_prev_exp / 'confusion_matrix.png'))
        #
        # plt.show()
        # plt.close('all')
        df_metrics = pd.DataFrame.from_dict(metrics_exp,
                                            orient='index',
                                            columns=['Results'])
        df_metrics.to_csv(str(self.path_prev_exp / 'metrics.csv'))

        return metrics_exp

    def get_path(self):
        return self.path_prev_exp

    def get_results(self):
        return self.results
