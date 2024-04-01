from typing import Any
import nibabel as nib
from pathlib import Path
from typing import overload
import numpy as np


class MRImageMask:
    @overload
    def __init__(self, other_mask: Any):
        self.__init__(other_mask.path)

    def __init__(self, path: Path):
        self._nifti_props = None
        self.path = path
        self._mask = None
        self._volume = None

    @property
    def data(self) -> np.ndarray:
        if self._mask is None:
            nifti_mask = nib.load(self.path)
            header = nifti_mask.header
            header.set_data_dtype(np.int32)
            self._mask = nifti_mask.get_fdata()
            self._nifti_props = {"header": header, "affine": nifti_mask.affine}
        return self._mask

    @property
    def volume(self) -> np.int32:
        if self._mask is not None:
            self._volume = np.sum(self._mask)
            return self._volume

    @property
    def nifti_props(self) -> dict:
        if self._nifti_props is None:
            _ = self.data
        return self._nifti_props
