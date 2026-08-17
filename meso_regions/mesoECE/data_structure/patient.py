import os
from types import NoneType
from typing import Union, Any
from pathlib import Path
import skimage
from meso_regions.mesoECE.data_structure.mrimage import MRImage


class Patient:
    def __init__(self,
                 path: Path,
                 path_masks: dict[str, Path],
                 id: int,
                 diagnosis: int = None,
                 subclass_diagnosis: str = None,
                 nodular: int = None
                 ):
        self._background_otsu = None
        self._path_masks = path_masks
        self._images = None
        self.time_points = None
        self.path = path
        self.id = id
        self.diagnosis = diagnosis
        self.subclass_diagnosis = subclass_diagnosis
        self.nodular = nodular
        self.supervoxels_m_masks = []
        self.supervoxels_b_masks = []
        self.load()

    def load(self) -> None:
        patient_dir = os.listdir(self.path)
        patient_file = [img for img in patient_dir if
                        MRImage.parse_name(img)[0] == self.id]

        # load list of time-points
        self.time_points = []
        for img in patient_file:
            _, t = MRImage.parse_name(img)
            self.time_points.append(t)
        self.time_points = sorted(self.time_points)

        # load list of images regarding time-points
        # and calculate its background
        self._background_otsu = {}
        self._images = {}
        for t in self.time_points:
            self._images[t] = MRImage(path=self.path,
                                      path_masks=self._path_masks,
                                      filename=MRImage.resolve_name(self.id, t))

            self._background_otsu[t] = skimage.filters.threshold_otsu(self._images[t].data)

    def get_image(self, t: int) -> MRImage:
        return self._images[t]

    def background_otsu(self, t: int) -> Union[NoneType, Any]:
        return self._background_otsu[t]

    @property
    def path_masks(self) -> dict[str, Path]:
        return self._path_masks
