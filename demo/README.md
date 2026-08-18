# Meso-Regions demo app (Gradio)

A small browser UI over one example case: scroll slices, isolate an ECE-positive
region and its curve, read the patient-level call.

```bash
pip install gradio
python demo/app.py
```

## Example data (not committed)

The app expects `demo/data/` with:

| File | Content |
| :-- | :-- |
| `ece_mask.nii.gz` | ECE-positive label map (region id per voxel) |
| `image.nii.gz` | reference volume (optional — falls back to the mask) |
| `curves.csv` | per-superspel mean-intensity curves (columns = time points) |

`demo/data/` is gitignored: example patient data — even derived masks — is governed by
PREDICT-Meso and must not be committed until its release is confirmed. Once a shareable
example exists, this app can be published as a Hugging Face Space with the data bundled.
