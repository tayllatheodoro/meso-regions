import pandas as pd
from pathlib import Path
from typing import List, Union
from tqdm import tqdm
from src.mesoECE.data_structure.diagnosis import Diagnosis
from src.mesoECE.data_structure.patient import Patient
from src.mesoECE.methods import AbstractMethod


class MesoDataset:
    def __init__(self,
                 path_images: Path,
                 path_classes: Path,
                 path_masks: dict[str, Path],
                 ids: Union[List[int], None] = None,
                 threads: int = 0
                 ):
        self.path_masks = path_masks
        self.patients: List[Patient] = []
        self.classes_df = None
        self.path_images = path_images
        self.path_classes = path_classes
        self.ids = ids
        self.threads = threads

        self.load()

    def load(self) -> None:
        # load diagnosis
        self.classes_df = pd.read_csv(self.path_classes)

        # load list of patients
        self.patients = []
        self.classes_df.set_index('ID', inplace=True)

        for id in tqdm(self.ids, desc="Loading Patients"):
            self.patients.append(Patient(path=self.path_images, id=id,
                                         diagnosis=self.classes_df.loc[
                                             id, 'CLASS'],
                                         subclass_diagnosis=self.classes_df.loc[
                                             id, 'SUBCLASS'],
                                         nodular=self.classes_df.loc[
                                             id, 'NODULAR'],
                                         path_masks=self.path_masks))

    def apply_method(self, method: AbstractMethod, **kwargs) -> None:

        if self.threads > 0 and method.thread_safe:
            from joblib import Parallel, delayed

            n = len(self.patients)
            patients_parallel_holder = [None] * n

            def process_patient(i):
                p = self.patients[i]
                patients_parallel_holder[i] = method.apply(p)
                return 0

            print(f"{method.__class__.__name__} processing started...")
            Parallel(n_jobs=self.threads, backend="threading", verbose=10)(
                delayed(process_patient)(i) for i in range(n))

            for i in range(n):
                self.patients[i] = patients_parallel_holder[i]
        else:
            for i, p in enumerate(self.patients):
                new_patient = method.apply(p)
                self.patients[i] = new_patient

    def get_diagnosed_patients(self, diagnosis: Diagnosis):
        pass
