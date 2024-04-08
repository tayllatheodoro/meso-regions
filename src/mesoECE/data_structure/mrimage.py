from pathlib import Path
from typing import Tuple
import nibabel as nib
import numpy as np
from src.mesoECE.data_structure.mrimage_mask import MRImageMask


class MRImage:
    def __init__(self, path: Path, path_masks: dict[str, Path], filename: str):
        self._nifti_props = None
        self.filename = filename
        self.path_masks = path_masks
        self.path = path
        self.masks: dict[str, MRImageMask] = {}
        self._image = None
        self._load_mask()
        self.img_path = self.path / filename

    @property
    def data(self) -> np.ndarray:
        if self._image is None:
            nifti_img = nib.load(self.img_path)
            header = nifti_img.header
            header.set_data_dtype(np.int32)
            self._nifti_props = {"header": header, "affine": nifti_img.affine}
            self._image = nifti_img.get_fdata()
        return self._image

    @property
    def nifti_props(self) -> dict:
        if self._nifti_props is None:
            _ = self.data
        return self._nifti_props

    def _load_mask(self) -> None:
        for masks_name, mask_dir in self.path_masks.items():
            self.masks[masks_name] = MRImageMask(mask_dir / self.filename)

    def get_mask(self, name: str) -> MRImageMask:
        if name in self.masks:
            return self.masks[name]

    @staticmethod
    def resolve_name(id, t, suffix="nii.gz") -> str:
        return f"{id:05d}_{t:03d}.{suffix}"

    @staticmethod
    def parse_name(name) -> Tuple[int,int]:
        id, t = name.split(".")[0].split("/")[-1].split("_")
        return int(id), int(t)
