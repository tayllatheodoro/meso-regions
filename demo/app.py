"""Gradio demo for Meso-Regions.

Interactive browser UI over one example case: scroll slices, isolate an
ECE-positive region and its signal-intensity curve, read the patient-level
call. Intended for demos (e.g. a Hugging Face Space with a bundled example);
for real analyses use the `meso-regions` CLI and HTML report.

Run locally:
    pip install gradio
    python demo/app.py

Expects example data in demo/data/ (not committed — see demo/README.md):
    ece_mask.nii.gz   ECE-positive label map (region id per voxel)
    image.nii.gz      reference volume (optional; falls back to the mask)
    curves.csv        per-superspel mean-intensity curves
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd

from meso_regions.report import (DISCLAIMER, PLANES, _is_ece_positive,
                                 _slice2d, _window_u8, region_color_map)

DATA_DIR = Path(__file__).parent / "data"
REF_T = 270
PLANE = "coronal"
AXIS = PLANES[PLANE]
ALPHA = 0.75


def load_case():
    mask = np.asanyarray(
        nib.load(str(DATA_DIR / "ece_mask.nii.gz")).dataobj).astype(np.int32)
    img_path = DATA_DIR / "image.nii.gz"
    if img_path.exists():
        image = np.asanyarray(nib.load(str(img_path)).dataobj).astype(float)
        if image.ndim > 3:
            image = image[..., 0]
    else:
        image = mask.astype(float)
    curves = pd.read_csv(DATA_DIR / "curves.csv")
    return _window_u8(image), mask, curves


IMAGE, MASK, CURVES = load_case()
COLOR_MAP = region_color_map(MASK)
T = [float(c) for c in CURVES.columns]
POSITIVE_ROWS = [i for i, (_, r) in enumerate(CURVES.iterrows())
                 if _is_ece_positive(T, r.values, REF_T)]
PRESENT = np.where((MASK > 0).sum(
    axis=tuple(i for i in range(3) if i != AXIS)) > 0)[0]
LO, HI = int(PRESENT.min()), int(PRESENT.max())
START = int(PRESENT[len(PRESENT) // 2])
CALL = ("ECE-POSITIVE — malignancy suspected by the automated rule"
        if POSITIVE_ROWS else "ECE-negative")


def region_choices():
    return ["all regions"] + [f"region {l}" for l in sorted(COLOR_MAP)]


def _selected_id(region):
    return int(region.split()[-1]) if region and region != "all regions" else 0


def render_slice(idx, region):
    sel = _selected_id(region)
    g = _slice2d(IMAGE, AXIS, int(idx))
    rgb = np.stack([g] * 3, axis=-1).astype(float)
    m = _slice2d(MASK, AXIS, int(idx))
    for lab, color in COLOR_MAP.items():
        if sel and lab != sel:
            continue
        c = np.array(color) * 255
        pix = m == lab
        rgb[pix] = rgb[pix] * (1 - ALPHA) + c * ALPHA
    return rgb.astype(np.uint8)


def render_curves(region):
    sel = _selected_id(region)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for i, (_, row) in enumerate(CURVES.iterrows()):
        if i in POSITIVE_ROWS and i in COLOR_MAP:
            dim = sel and i != sel
            ax.plot(T, row.values, color=COLOR_MAP[i],
                    lw=2.0 if not dim else 1.0,
                    alpha=0.15 if dim else 1.0, zorder=2)
            if not dim:
                ax.annotate(str(i), (T[-1], row.values[-1]), xytext=(4, 0),
                            textcoords="offset points", fontsize=7,
                            color=COLOR_MAP[i], va="center")
        else:
            ax.plot(T, row.values, color="0.85", lw=0.8, zorder=1)
    ax.axvline(REF_T, color="0.4", ls="--", lw=1,
               label=f"{REF_T} s reference")
    ax.set_xlabel("Time post-contrast (s)")
    ax.set_ylabel("Mean signal intensity")
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def update(idx, region):
    return render_slice(idx, region), render_curves(region)


def main():
    import gradio as gr

    with gr.Blocks(title="Meso-Regions demo") as demo:
        gr.Markdown(f"""# Meso-Regions — interactive demo
Automated 4D detection of the Early Contrast Enhancement (ECE) biomarker in
pleural DCE-MRI ([paper](https://github.com/tayllatheodoro/meso-regions#citation)).

**{CALL}**

> ⚠️ {DISCLAIMER}
""")
        with gr.Row():
            with gr.Column():
                img = gr.Image(render_slice(START, "all regions"),
                               label=f"{PLANE} slice", interactive=False)
                idx = gr.Slider(LO, HI, value=START, step=1,
                                label=f"{PLANE} slice")
            with gr.Column():
                plot = gr.Plot(render_curves("all regions"),
                               label="signal-intensity curves")
                region = gr.Dropdown(region_choices(), value="all regions",
                                     label="isolate ECE region "
                                           "(colors match the image)")
        idx.change(update, [idx, region], [img, plot])
        region.change(update, [idx, region], [img, plot])
    demo.launch()


if __name__ == "__main__":
    main()
