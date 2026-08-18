---
name: run-meso-regions
description: Run the Meso-Regions pleural DCE-MRI pipeline end to end — DICOM ingestion, YAML config, pipeline execution, and HTML patient reports. Use when the user wants to analyse DCE-MRI data with meso-regions, convert DICOM to NIfTI for it, or generate/view patient reports.
---

# Running the Meso-Regions pipeline

Meso-Regions detects Early Contrast Enhancement (ECE) biomarker regions in 4D pleural
DCE-MRI. **Research use only — never present outputs as clinical diagnosis.**

## Workflow

1. **Install** (once): `pip install -e .` from the repo root. DISF supervoxels additionally
   need libIFT (`disf` stage `ift_path`); SLIC works out of the box.

2. **Ingest DICOM** (if the user has DICOM, e.g. from PACS):
   ```bash
   meso-regions convert --dicom-dir <root_of_series_folders> -o <nifti_dir> \
       --name-pattern '(\d+)_(\d+)'   # if folders are named <patient>_<time>
   ```
   The pipeline expects one NIfTI per patient per time point, named `<patient>_<time>.nii.gz`,
   with time points in seconds post-contrast (0, 40, 80, 180, 270, 540, 810 in the original
   study). Ask the user which series are which if folder names don't encode it.

3. **Configure**: `meso-regions example-config > config.yaml`, then edit paths (images,
   masks, classes CSV, output), patient ids, and the stage list. `meso-regions stages`
   lists all stages and their options. Typical order:
   `resample` → `ants_reg` → `dilate` → `slic` (or `disf`) → `full_ece`.
   Pleural-effusion masks are a required input (semi-automatic step — the user provides
   them; there is no automated effusion segmentation yet).

4. **Validate before running**: `meso-regions run -c config.yaml --dry-run` — always do
   this first and show the user the resolved config.

5. **Run**: `meso-regions run -c config.yaml`. Long; run in background and report metrics
   when done.

6. **Report** (per patient):
   ```bash
   meso-regions report --image <t270.nii.gz> --fluid-mask <fluid.nii.gz> \
       --ece-mask <ece_labels.nii.gz> --curves <mean_intensity.csv> \
       --patient-id <id> --method DISF -o report_<id>.html
   ```
   The HTML is self-contained (no data leaves the machine). Open it in a browser.
   For 3D inspection suggest ITK-SNAP or 3D Slicer (masks are standard NIfTI), or the
   Gradio demo (`python demo/app.py`).

## Experimental options

- `meso_regions.experimental.dynamic_band` — enhancement-guided pleural band (mask v2
  prototype); see docs/ROADMAP.md Phase 2 before using.

## Gotchas

- Patient ids are integers in classes/splits CSVs but zero-padded in file names (`00325`).
- All volumes for one patient must share a grid before `resample`.
- dcm2niix may exit non-zero on warnings while still converting correctly.
