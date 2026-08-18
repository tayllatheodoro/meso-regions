"""EXPERIMENTAL: enhancement-guided dynamic pleural band (mask v2 prototype).

Status: validated through an expert-in-the-loop prototype on 2 patients
(BAPE 325, sarcomatoid 14) in Aug 2026; NOT used in any published results.
The published pipeline uses the fixed effusion-dilation band. See
docs/ROADMAP.md Phase 2 for design rationale and validation notes.

Usage: python build_dynamic_band.py <pid5> <ts_dir> <reg_t0> <fwd_warp> <fwd_affine> <fluid_gt>
Writes masks + itksnap label map to masks_<pid>/ next to this script.

v4 logic: lungs∪fluid surface -> distance shells (cap 8 = ~7 mm allowance) ->
fine sectors (24 azimuth x 4 vertical bands per hemithorax) -> per-sector
enhancement profile from t270−t0reg. Extent = decay endpoint
(base+0.35*(peak−base)); if it never triggers, extent = peak+2 (NOT the cap):
thick only where enhancement justifies it. Voxel gate: within allowed shells a
voxel needs subtraction signal >= sector baseline (rejects dark rib cortex).
Organ exclusion: heart (wide guard) + abdominal organs + spine + scapulae.
"""
import os, sys
import ants
import nibabel as nib
import numpy as np
from scipy.ndimage import binary_dilation, binary_fill_holes, distance_transform_edt

pid, ts_dir, reg_t0, warp_f, aff_f, fluid_f = sys.argv[1:7]
lungs_override = sys.argv[7] if len(sys.argv) > 7 else None  # e.g. CT->MRI transferred lungs
D = "/Users/taylla.theodoro/Downloads/meso-regions/raw_images_dicom_to_nii 2.gz"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"masks_{pid}")
os.makedirs(OUT, exist_ok=True)

fix = ants.image_read(f"{D}/{pid}_270.nii.gz")
tx = [warp_f, aff_f]

def warp(name):
    p = f"{ts_dir}/{name}.nii.gz"
    if not os.path.exists(p):
        return None
    return ants.apply_transforms(fixed=fix, moving=ants.image_read(p),
                                 transformlist=tx,
                                 interpolator="nearestNeighbor").numpy() > 0

if lungs_override:
    lungs = binary_fill_holes(np.asanyarray(nib.load(lungs_override).dataobj) > 0)
    mid = lungs.shape[0] // 2
    _ix = np.indices(lungs.shape)[0]
    lr = lungs & (_ix < mid)
    ll = lungs & (_ix >= mid)
else:
    ll = warp("lung_left"); lr = warp("lung_right")
    lungs = binary_fill_holes(ll | lr)
fluid = np.asanyarray(nib.load(fluid_f).dataobj) > 0
vox_mm = float(nib.load(fluid_f).header.get_zooms()[0])
HEART_GUARD_MM, ORGAN_GUARD_MM = 10.0, 2.0        # guards in mm, not voxels
excl = binary_dilation(warp("heart"),
                       iterations=max(1, round(HEART_GUARD_MM / vox_mm)))
for org in ["liver", "spleen", "stomach", "kidney_left", "kidney_right",
            "aorta", "esophagus", "vertebrae", "spinal_cord",
            "intervertebral_discs", "autochthon_left", "autochthon_right",
            "scapula_left", "scapula_right", "inferior_vena_cava"]:
    w = warp(org)
    if w is not None:
        excl |= binary_dilation(w, iterations=max(1, round(ORGAN_GUARD_MM / vox_mm)))
t270 = fix.numpy().astype(np.float32)
t0 = ants.image_read(reg_t0).numpy().astype(np.float32)
S = np.clip(t270 - t0, 0, None)

U = lungs | fluid
dist = distance_transform_edt(~U).astype(np.float32)
NSH, CAP = 10, 8
shell = np.where((dist > 0) & (dist <= NSH), np.ceil(dist), 0).astype(np.int16)
# exclusion limits outward growth but may never remove the 2 innermost shells:
# the pleura lies against the lung/fluid surface by definition
shell[excl & (dist > 2)] = 0

X, Y, Z = np.indices(S.shape, dtype=np.int16)
side = (X >= S.shape[0] // 2).astype(np.int16)
cx = {0: np.array(np.where(lr)).mean(1), 1: np.array(np.where(ll)).mean(1)}
NAZ, NY = 24, 4
band = shell > 0
ys = Y[band]
ybands = np.clip(((Y - ys.min()) / (max(ys.max() - ys.min(), 1) / NY)).astype(np.int16), 0, NY - 1)
sector = np.zeros_like(shell)
for s in (0, 1):
    m = band & (side == s)
    ang = ((np.arctan2(Z[m] - cx[s][2], X[m] - cx[s][0]) + np.pi) / (2 * np.pi) * NAZ).astype(np.int16) % NAZ
    sector[m] = 1 + s * NAZ * NY + ybands[m] * NAZ + ang

dyn = np.zeros_like(band)
exts = {}
for sec in np.unique(sector[sector > 0]):
    msec = sector == sec
    p = np.array([S[msec & (shell == k)].mean() if (msec & (shell == k)).any() else 0
                  for k in range(1, NSH + 1)])
    if p.max() <= 0:
        continue
    pk = int(np.argmax(p[:CAP]))
    base = np.median(p[-3:])
    th = base + 0.35 * (p[pk] - base)
    ext = 0
    for k in range(pk + 1, CAP + 1):
        if k < len(p) and p[k] <= th:
            ext = k
            break
    if ext == 0:
        ext = min(pk + 2, CAP)          # no decay found -> modest, not cap
    ext = int(np.clip(ext, 2, CAP))
    exts[int(sec)] = ext
    gate = S >= base                     # voxel-level enhancement gate
    dyn |= msec & (shell <= ext) & gate
dyn |= (shell > 0) & (shell <= 2)        # always keep the 2 innermost shells

aff = nib.load(fluid_f).affine
for name, arr in [("dynamic_band", dyn), ("lungs_filled", lungs),
                  ("organ_exclusion", excl),
                  ("fixed_band", binary_dilation(fluid, iterations=2) & ~fluid)]:
    nib.save(nib.Nifti1Image(arr.astype(np.uint8), aff), f"{OUT}/{name}.nii.gz")
nib.save(nib.Nifti1Image(S, aff), f"{OUT}/subtraction_t270_t0.nii.gz")
lab = np.zeros(fluid.shape, dtype=np.uint8)
lab[lungs] = 3; lab[excl] = 4; lab[dyn] = 2; lab[fluid] = 1
nib.save(nib.Nifti1Image(lab, aff), f"{OUT}/itksnap_labels.nii.gz")
e = np.array(list(exts.values()))
print(f"{pid} v4: band {int(dyn.sum())} vox ({dyn.sum()*vox_mm**3/1000:.0f} mL) | "
      f"extents mean {e.mean():.1f} min {e.min()} max {e.max()} (cap {CAP}) | "
      f"thickness {e.mean()*vox_mm:.1f} mm mean, up to {e.max()*vox_mm:.1f} mm")
