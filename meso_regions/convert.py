"""DICOM -> NIfTI ingestion for Meso-Regions.

Converts a directory of DICOM series (one sub-directory per series, as
exported from PACS) into the per-patient, per-time-point NIfTI layout the
pipeline expects (``<patient>_<time>.nii.gz``).

Uses dcm2niix when available (recommended; ``pip install dcm2niix``), with a
SimpleITK fallback. No data leaves the machine.
"""

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


def _dcm2niix_bin():
    exe = shutil.which("dcm2niix")
    if exe:
        return exe
    try:  # pip wheel exposes the binary through the package
        import dcm2niix
        return dcm2niix.bin
    except ImportError:
        return None


def _convert_series_dcm2niix(exe, series_dir, out_dir, name):
    # dcm2niix exits non-zero on warnings while still writing valid output,
    # so success is judged by the presence of the output file
    r = subprocess.run([exe, "-z", "y", "-f", name, "-o", str(out_dir),
                        str(series_dir)], capture_output=True, text=True)
    if not list(Path(out_dir).glob(f"{name}*.nii.gz")):
        raise RuntimeError(f"dcm2niix produced no output: "
                           f"{r.stderr.strip()[-200:] or r.stdout.strip()[-200:]}")


def _convert_series_sitk(series_dir, out_dir, name):
    import SimpleITK as sitk
    reader = sitk.ImageSeriesReader()
    ids = reader.GetGDCMSeriesIDs(str(series_dir))
    if not ids:
        raise RuntimeError(f"no DICOM series found in {series_dir}")
    reader.SetFileNames(reader.GetGDCMSeriesFileNames(str(series_dir), ids[0]))
    sitk.WriteImage(reader.Execute(), str(out_dir / f"{name}.nii.gz"))


def convert_dicom_tree(dicom_root, out_dir, pattern=None):
    """Convert every DICOM series directory under ``dicom_root``.

    Output name per series: taken from the series directory name (sanitised);
    if ``pattern`` (a regex with two groups: patient, time) matches the
    directory name, the canonical ``<patient>_<time>`` name is used.
    Returns list of written stems.
    """
    dicom_root, out_dir = Path(dicom_root), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    exe = _dcm2niix_bin()

    series_dirs = sorted({p.parent for p in dicom_root.rglob("*")
                          if p.is_file() and (p.suffix.lower() in
                                              (".dcm", ".ima") or
                                              p.name.upper().startswith("MR"))})
    if not series_dirs:
        sys.exit(f"no DICOM files found under {dicom_root}")

    written = []
    for sd in series_dirs:
        name = re.sub(r"[^\w\-]", "_", sd.name)
        if pattern:
            m = re.search(pattern, sd.name)
            if m:
                name = f"{m.group(1)}_{m.group(2)}"
        try:
            if exe:
                _convert_series_dcm2niix(exe, sd, out_dir, name)
            else:
                _convert_series_sitk(sd, out_dir, name)
            written.append(name)
            print(f"  {sd.name} -> {name}.nii.gz")
        except Exception as e:  # keep going; report at the end
            print(f"  FAILED {sd.name}: {e}", file=sys.stderr)
    if not exe:
        print("note: converted with SimpleITK fallback; for robust handling "
              "of vendor DICOM install dcm2niix (pip install dcm2niix)")
    return written
