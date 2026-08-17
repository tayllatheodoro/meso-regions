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

The pipeline stages are driven by configuration dictionaries (`meso_regions/run/config.py`)
and stage scripts under `meso_regions/run/`:

```bash
python -m meso_regions.run.run_reg        # motion correction
python -m meso_regions.run.run_dilation   # pleural-region mask
python -m meso_regions.run.run_slic       # SLIC superspels (or run_disf)
python -m meso_regions.run.run_full_ece   # curves + ECE rule + metrics
```

Inputs are NIfTI volumes organised per patient and time point. See `meso_regions/run/config.py`
for the expected directory layout and stage options. A single-command CLI is planned
(see [ROADMAP](docs/ROADMAP.md)).

## Data

The imaging dataset used in the papers is governed by the PREDICT-Meso network. Researchers can
request access at: https://www.predictmeso.com/research-tissue-bank/#rtbform. The curated 4D
pleural-effusion ground-truth masks are released alongside the journal paper.

## Citation

If you use this software, please cite:

> Theodoro TM, Silva IF, Tsim S, Blyth K, Falcão AX. *Meso-Regions: 4D segmentation of early
> contrast enhancement biomarkers for pleural mesothelioma diagnosis.* Proc IEEE 23rd
> International Symposium on Biomedical Imaging (ISBI), 2026.

The ECE biomarker was originally described in:

> Tsim S, et al. *Early contrast enhancement: a novel magnetic resonance imaging biomarker of
> pleural malignancy.* Lung Cancer 2018;118:48–56.

## License

GPL-3.0 — see [LICENSE](LICENSE).
