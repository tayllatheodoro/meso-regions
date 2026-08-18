# Meso-Regions

Automated 4D segmentation of **Early Contrast Enhancement (ECE)** biomarker regions in pleural
DCE-MRI, for pleural mesothelioma research.

Meso-Regions takes a 4D (3D + time) dynamic contrast-enhanced MRI series and produces
patient-specific ECE-positive segmentation masks and a patient-level malignancy call, without
manual ROI placement:

1. **Motion correction** — SyN diffeomorphic registration (ANTs) of all time points to the
   270-s post-contrast reference.
2. **Pleural region extraction** — semi-automatic pleural-effusion segmentation (image foresting
   transform) followed by morphological rim isolation around the fluid.
3. **4D supervoxels ("superspels")** — SLIC or DISF partitions of the pleural region, propagated
   across all time points.
4. **Signal-intensity descriptors** — background-corrected mean-intensity curve per superspel.
5. **ECE rule classification** — a superspel is ECE-positive if its curve peaks at ≤270 s, has its
   minimum pre-contrast, and rises monotonically to the peak then falls; a patient is called
   malignant if any superspel is ECE-positive.

![Meso-Regions pipeline overview](docs/figures/overview.png)
*Pipeline overview — input DCE-MRI, registration, pleural-region mask generation, per-superspel
intensity descriptors, and ECE-rule classification. Figure from Theodoro et al., IEEE ISBI 2026
(© 2026 IEEE).*

![Representative workflow on one patient](docs/figures/figure_workflow_patient116.png)
*Representative coronal workflow at 270 s post-contrast: native image, segmented pleural
effusion, derived pleural-region mask, DISF supervoxels, and ECE-positive output (panels D–E
from Theodoro et al., IEEE ISBI 2026, © 2026 IEEE).*

![Intensity curves by histology](docs/figures/curve_shape_malignant_vs_benign.png)
*Patient-mean peak-normalised signal-intensity curves stratified by histology (shaded bands:
SEM), for SLIC and DISF partitions.*

> ⚠️ **Research use only.** This software is not a medical device and must not be used for
> clinical diagnosis or treatment decisions.

## Installation

```bash
git clone https://github.com/tayllatheodoro/meso-regions.git
cd meso-regions
pip install -e .
```

DISF supervoxels additionally require the
[libIFT / DISF reference implementation](https://github.com/LIDS-UNICAMP) (LIDS-UNICAMP);
SLIC works out of the box via scikit-image.

## Usage

### From DICOM

Doctors and researchers can start directly from exported DICOM series:

```bash
meso-regions convert --dicom-dir /path/to/dicom_series -o nifti/ --name-pattern '(\d+)_(\d+)'
```

This produces the `<patient>_<time>.nii.gz` layout the pipeline expects (uses dcm2niix,
`pip install dcm2niix`; SimpleITK fallback). All processing is local — no data upload.

### Pipeline

The pipeline is driven by a YAML config through the `meso-regions` CLI:

```bash
meso-regions stages                      # list available pipeline stages + options
meso-regions example-config > config.yaml
# edit config.yaml: paths, patient ids, pipeline stages
meso-regions run -c config.yaml --dry-run   # validate without executing
meso-regions run -c config.yaml
```

A pipeline is an ordered list of stages (`resample`, `ants_reg`, `dilate`, `slic`, `disf`,
`full_ece`, …); each stage's options map to the functions in `meso_regions/run/config.py`.
Inputs are NIfTI volumes organised per patient and time point. Note: the `disf` stage's
`ift_path` must point to your local libIFT build. The original research scripts remain
under `meso_regions/run/` for reference.

### Patient report (visualization)

Generate a self-contained HTML report per patient — slice overlays (label-map layers such as
supervoxels and ECE regions are drawn with one color per region), an interactive slice slider
(drag or mouse-wheel), per-superspel signal-intensity curves with the ECE rule applied, and the
patient-level call. Everything is embedded in one file: no install needed to view it, and no
data leaves the machine.

```bash
meso-regions report \
  --image patient_270s.nii.gz \
  --fluid-mask fluid.nii.gz --pleural-mask pleural_region.nii.gz \
  --supervoxels disf_labels.nii.gz --ece-mask ece_mask.nii.gz \
  --curves mean_intensity.csv \
  --patient-id 116 --method DISF -o report_116.html
```

Use `--scroll-plane axial|coronal|sagittal` to choose the slider plane and `--no-scroll` for a
smaller file without the slider.

### Viewing outputs in existing tools

All masks are standard NIfTI, so they load directly as segmentation overlays in
[3D Slicer](https://www.slicer.org) or [ITK-SNAP](http://www.itksnap.org): open the reference
volume, then add the ECE / pleural-region mask as a segmentation/overlay layer.

## Data

The imaging dataset used in the papers is governed by the PREDICT-Meso network. Researchers can
request access at: https://www.predictmeso.com/research-tissue-bank/#rtbform. The curated 4D
pleural-effusion ground-truth masks are released alongside the journal paper.

## Citation

If you use this software, please cite the ISBI 2026 paper:

> Theodoro TM, Silva IF, Tsim S, Blyth K, Falcão AX. *Meso-Regions: 4D segmentation of early
> contrast enhancement biomarkers for pleural mesothelioma diagnosis.* Proc IEEE 23rd
> International Symposium on Biomedical Imaging (ISBI), 2026.

```bibtex
@inproceedings{theodoro2026mesoregions,
  author    = {Theodoro, Taylla M. and Silva, Ilan F. and Tsim, Selina and
               Blyth, Kevin and Falc{\~a}o, Alexandre X.},
  title     = {Meso-Regions: {4D} Segmentation of Early Contrast Enhancement
               Biomarkers for Pleural Mesothelioma Diagnosis},
  booktitle = {Proc. IEEE 23rd International Symposium on Biomedical Imaging (ISBI)},
  year      = {2026},
}
```

### Related publications

The method was first presented at the 16th International Mesothelioma Interest Group (iMig)
conference and covered in its imaging review:

> Armato SG III, Katz SI, Frauenfelder T, Jayasekera G, Catino A, Blyth KG, Theodoro T,
> Rousset P, Nackaerts K, Opitz I. *Imaging in pleural mesothelioma: a review of the 16th
> international conference of the International Mesothelioma Interest Group.* Lung Cancer
> 2024;193:107832.

The ECE biomarker was originally described in:

> Tsim S, Humphreys CA, Cowell GW, Stobo DB, Noble C, Woodward R, Kelly CA, Alexander L,
> Foster JE, Dick C, Blyth KG. *Early contrast enhancement: a novel magnetic resonance imaging
> biomarker of pleural malignancy.* Lung Cancer 2018;118:48–56.

Extended methodology in the master's dissertation:

> Theodoro TM. *Towards Automatic Detection of Pleural Mesothelioma Biomarker in 4D Dynamic
> MR Imaging.* MSc dissertation, Institute of Computing, University of Campinas (UNICAMP).
> https://github.com/tayllatheodoro/meso-master-thesis

## License

GPL-3.0 — see [LICENSE](LICENSE).
