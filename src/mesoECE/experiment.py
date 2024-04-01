import shutil

import json
import os
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
from sklearn import metrics

import numpy as np
import pandas as pd

import mesexp.methods.registration as registration_module
import mesexp.methods.preprocessing as preprocessing_module
import mesexp.methods.supervoxel as supervoxel_module
import mesexp.methods.classifier as classifier_module
import MesoDataset


class Experiment:
    def __init__(self,
                 experiment_name: str,
                 path_images: Path,
                 path_masks: Path,
                 path_experiments: Path,
                 path_classes: Path,
                 ids: List[int],
                 config: dict,
                 threads: int = 0):
        self.results = None
        self.config = config
        self.path_experiments = path_experiments / experiment_name
        shutil.copytree(path_masks, self.path_experiments / 'masks' / 'pleural_region', dirs_exist_ok=True)

        os.makedirs(self.path_experiments, exist_ok=True)

        self.pipeline_methods = []

        self._instantiate_experiment()

        masks_dir = {'pleural_region': path_masks}

        self.image_dataset = MesotheliomaDataset(path_patients=path_images, path_classes=path_classes,
                                                 path_masks=masks_dir, ids=ids, threads=threads)

    def _parse_experiment_partial_files(self,
                                        config: dict) -> Path:
        """
        Return the path from the experiment containing the desired configuration or
        create a new path which will be the output for the operations under your
        configuration.
        """
        path_partial_experiments = self.path_experiments / config["method"]

        os.makedirs(path_partial_experiments, exist_ok=True)

        partial_experiments_dirs = sorted(os.listdir(path_partial_experiments))

        if len(partial_experiments_dirs) > 0:
            # try to read the experiment config and data
            for part_exp in partial_experiments_dirs:
                with open(path_partial_experiments / part_exp / 'config.json') as json_file:
                    current_config = json.load(json_file)
                    if current_config == config:
                        return path_partial_experiments / part_exp
            id_exp = int(partial_experiments_dirs[-1]) + 1
        else:
            id_exp = 1

        # create a new experiment dir
        new_dir = f"{id_exp:05d}"
        try:
            os.makedirs(path_partial_experiments / new_dir)
        except:
            pass
        json_out_file_path = path_partial_experiments / new_dir / 'config.json'

        with open(json_out_file_path, "w") as json_file:
            json.dump(config, json_file)
        return path_partial_experiments / new_dir

    def _instantiate_experiment(self):
        config_list = self.config

        mapping_module = {
            "registration": registration_module,
            "supervoxel": supervoxel_module,
            "preprocessing": preprocessing_module,
            "classifier": classifier_module
        }

        self.path_previous_experiment = ""
        self.pipeline_methods = []

        for config in config_list:
            method_module = config["module"]
            method_name = config["method"]
            args = config["args"]
            config["previous_experiment"] = str(self.path_previous_experiment)


            module = mapping_module[method_module]
            method_class = getattr(module, method_name)

            path_partial_experiment = self._parse_experiment_partial_files(config)

            instance = method_class(path=path_partial_experiment, **args)

            self.pipeline_methods.append(instance)
            self.path_previous_experiment = path_partial_experiment
            # save config in a file for all methods

            print(config)

            with open(path_partial_experiment / 'config_all.json', "w") as json_file:
                json.dump(config_list, json_file)

    def execute_pipeline(self):
        self.results = {}
        for method in self.pipeline_methods:
            self.image_dataset.apply_method(method)
            if method.results() is not None:
                self.results[method.__class__.__name__] = method.results()
        return self.results

    def results_to_pandas(self):

        # check if ECE is in the methods
        self.classifier =""
        if 'ECE' in self.results:
            self.classifier = "ECE"
        elif 'DividingECE' in self.results:
            self.classifier = "DividingECE"
        elif 'FullECE' in self.results:
            self.classifier = "FullECE"

        df = pd.DataFrame(self.results[self.classifier],
                          columns=['IDs', 'GRUND_TRUTH', 'PREDICT', 'SUBTYPE', 'NODULAR', 'QTDE_SUPERVOXELS', 'VOLUME_SUPERVOXELS'])
        path_results = self.path_previous_experiment / 'diagnosis.csv'
        df.to_csv(str(path_results), index=False)

    def results_seeds(self):
        supervoxel =""
        if 'SLIC' in self.results:
            supervoxel = "SLIC"
        elif'DISF' in self.results:
            supervoxel = "DISF"
        elif 'SICLE' in self.results:
            supervoxel = "SICLE"

        df = pd.DataFrame(self.results[supervoxel],
                          columns=['IDs', 'INITIAL_SEEDS','FINAL_SEEDS'])
        path_results = self.path_previous_experiment / 'supervoxel_n_seeds.csv'
        df.to_csv(str(path_results), index=False)

    def results_final_seeds(self):
        supervoxel =""
        if 'SLIC' in self.results:
            supervoxel = "SLIC"
        elif'DISF' in self.results:
            supervoxel = "DISF"
        elif 'SICLE' in self.results:
            supervoxel = "SICLE"

        df = pd.DataFrame(self.results[supervoxel],
                          columns=['IDs', 'FINAL_SEEDS'])
        path_results = self.path_previous_experiment / 'final_seeds.csv'
        df.to_csv(str(path_results), index=False)


    def metrics(self):

        results_ece = self.results[self.classifier]
        metrics_exp = {}
        FP = []
        FN = []

        y = [result_ece[1] for result_ece in results_ece]
        y_pred = [result_ece[2] for result_ece in results_ece]

        # add false positive cases and IDS to a FP
        for result_ece in results_ece:
            if result_ece[1] == 1 and result_ece[2] == 0:
                FN.append(result_ece[0])
            if result_ece[1] == 0 and result_ece[2] == 1:
                FP.append(result_ece[0])

        metrics_exp['acc'] = metrics.accuracy_score(y, y_pred)
        metrics_exp['balanced_acc'] = metrics.balanced_accuracy_score(y, y_pred)
        metrics_exp['jaccard'] = metrics.jaccard_score(y, y_pred)
        metrics_exp['f1_score'] = metrics.f1_score(y, y_pred)
        metrics_exp['average_precision_score'] = metrics.precision_score(y, y_pred)
        metrics_exp['AUC'] = metrics.roc_auc_score(y, y_pred)
        metrics_exp['Sensitivity'] = metrics.recall_score(y, y_pred)
        metrics_exp['Specificity'] = metrics.precision_score(y, y_pred)

        tn, fp, fn, tp = metrics.confusion_matrix(y, y_pred).ravel()
        metrics_exp['FP'] = fp
        metrics_exp['FN'] = fn
        metrics_exp['TN'] = tn
        metrics_exp['TP'] = tp

        metrics_exp['PPV'] = tp / (tp + fp)
        metrics_exp['NPV'] = tn / (tn + fn)
        metrics_exp['FP_ID'] = FP
        metrics_exp['FN_ID'] = FN

        roc_display = metrics.RocCurveDisplay.from_predictions(y, y_pred)
        roc_display.plot()
        plt.savefig(self.path_previous_experiment / 'roc_curve.png')
        plt.show()

        # save confusion matrix with sklean

        cm = metrics.confusion_matrix(y, y_pred, normalize='all')
        cmd = metrics.ConfusionMatrixDisplay(cm, display_labels=['NON-MALIGNANT', 'MALIGNANT'])
        cmd.plot()
        cmd.figure_.savefig(str(self.path_previous_experiment / 'confusion_matrix.png'))
        plt.show()

        df = pd.DataFrame.from_dict(metrics_exp, orient='index', columns=['Results'])
        df.to_csv(str(self.path_previous_experiment / 'metrics.csv'))

        return metrics_exp

    def get_path(self):
        return self.path_previous_experiment

    def get_results(self):
        return self.results