# Roadmap

Goal: evolve Meso-Regions from research code into a reproducible open-source tool for the
pleural-imaging community. The published ISBI 2026 pipeline stays as the reference baseline;
every upgrade below must be benchmarked against it on the same 3 data splits.

## Phase 1 — installable & reproducible (in progress)

- [x] Proper package layout (`meso_regions`), `pyproject.toml`, pinned dependencies
- [x] README with pipeline overview, install, research-use-only disclaimer, citation
- [ ] Single-command CLI (`meso-regions run --config config.yaml`) replacing per-stage scripts
- [ ] Config file (YAML) instead of hard-coded paths in `run/config.py`
- [ ] Smoke test on one synthetic/phantom patient + CI (GitHub Actions)
- [ ] Document DISF/libIFT build steps or vendor a pip-installable binding

## Phase 2 — pleural mask v2: enhancement-guided dynamic rim

The current mask (fixed 2-voxel effusion dilation minus fluid) has two failure modes observed
by T.M.T.: pleural lumps around the lungs outside the band, and non-pleura inside it.

**Design (2026-08-18), adapting the layer/section skin-segmentation approach of Gallegos et
al., ISBI 2026 poster 1571238881 (mammographic skin thickness for IBC):**

1. **Reference surface** — hole-filled TotalSegmentator-MRI lungs (validated 2026-08-18 on
   patient 116: 30 s/patient CPU, correct even with collapsed lung; see
   `paper3_exploration/ts116_lungs_pilot.png`) ∪ effusion surface (existing fluid masks).
2. **Enhancement signal** — subtraction image t270 − t0 on the registered stack (pleura and
   other vascular tissue are near-invisible at t=0 and highlighted at t>0 — T.M.T.).
3. **Dynamic rim** — 1-voxel distance shells marching outward from the surface, per surface
   sector; the local band extent ends where the per-sector subtraction-signal profile decays
   past its peak (with a max-thickness cap to reject enhancing blobs like vessels/heart).
4. **Byproduct biomarker** — per-sector pleural THICKNESS map (pleural thickening is an
   established malignancy criterion on CT; an MRI thickness map is novel, interpretable output).
5. **Anatomy-aware sectors (T.M.T., 2026-08-18)** — mesothelioma is usually unilateral and
   thickening is predominantly costal (rib-adjacent), not mediastinal; the heart is a
   high-flow FP source. Therefore: (a) process hemithoraces separately and use the
   contralateral side as internal control (per-sector asymmetry score normalises scanner /
   dose / physiology — analogous to IBC vs contralateral breast in Gallegos et al.);
   (b) label sectors costal / diaphragmatic / mediastinal — costal carries diagnostic
   weight, mediastinal down-weighted (not excluded: rare true cardiac-adjacent thickening
   exists); (c) explicit heart + great-vessel exclusion mask from TotalSegmentator-MRI.
6. **Validation** — coverage vs the published band, overlap with expert ECE regions, effect on
   the 3-split metrics; registration quality is an explicit dependency (subtraction).
- [x] Single-patient prototype (325, BAPE): pleural signature confirmed, organ exclusion
      works, band covers both hemithoraces — see paper3_exploration/FINDINGS.md (2026-08-18)
- [ ] Fine surface-patch sectors (~hundreds) to tighten extents off the cap
- [ ] Batch the ~31 patients with local native volumes + fluid masks
- [ ] Adaptive per-sector extent replaces the fixed dilation radius entirely
- [ ] Retro-check: are existing ECE false positives heart-adjacent? (TS heart mask vs
      ECE-positive region locations — cheap and informative for the current paper's Discussion)

## Phase 3 — NEXT UP: clustering + radiomics with existing masks (paper 3)

Agreed scope (2026-08-17) — one contained study, every input already on disk, nothing new
built. Ships after the RCTI paper is submitted.

1. **DISF regions only** (SLIC regions ~330 voxels — too small for texture statistics);
   existing pleural masks and the published 3 splits.
2. **Features per superspel**: existing mean-intensity curves + existing std-curves
   (within-region heterogeneity) + PyRadiomics (IBSI-standard settings) on the registered
   volumes inside each region. Registered-domain interpolation stated as a limitation,
   not made a prerequisite study.
3. **Enhancement-phenotype clustering**: k-means (k≈4–6) on normalized trajectories →
   per-patient composition vectors ("habitat" style).
4. **One simple model**: regularized logistic regression on composition + ECE burden
   (fraction of rule-positive superspels), grouped CV on the same splits.
5. **One question**: does specificity improve over the mean-intensity ECE rule,
   especially in BAPE?

**Status 2026-08-18 — curve-based arm complete, three negatives (see
`meso-regions-results/paper3_exploration/FINDINGS.md`):** phenotype-composition model at
chance (AUC 0.46); ECE-burden refuted (BAPE median burden 0.071 ≈ mesothelioma 0.068 — benign
inflamed pleura enhances fractionally like tumour); std-curve heterogeneity uninformative
(AUC 0.52). Region count predicts (AUC 0.69) but is an effusion-size confound. Conclusion:
mean-intensity dynamics are exhausted at region AND patient level; the standing candidates
are texture radiomics on the volumes (needs full registered volumes from Drive/lab) and the
CT-informed pleural mask.

Groundwork already done: superspel curve-shape t-SNE over all 56 patients
(`meso-regions-results/superspel_embedding.csv`) shows malignant/benign overlap in
curve-shape space — the motivation for texture features.

## Phase 3a — external validation on open breast DCE-MRI

An open DCE-MRI dataset as comparison/benchmark for the follow-up study. Candidates:
MAMA-MIA (~1,500 cases with expert primary-tumour segmentations, aggregated from
Duke / ISPY1 / ISPY2 / NACT), Duke-Breast-Cancer-MRI (TCIA, 922 patients),
QIN Breast DCE-MRI (TCIA). Value:

- **Generalizability**: supervoxel + enhancement-trajectory analysis on a different
  organ/cancer with known tumour masks — tests that the method is not pleura-specific.
- **Scale**: radiomics/clustering experiments at n≫56 before applying them to the
  pleural cohort.
- **Reproducibility**: a fully public end-to-end example for the repo — and a
  governance-free bundled case for the Gradio demo / Hugging Face Space.

## Phase 3b — parked (not abandoned; revisit after paper 3)

- **CT-informed pleural mask**: Kevin Blyth's CT pleural segmentation on paired patients —
  first quantify how much true pleura the effusion-dilation band misses (CT→MRI transfer,
  `utils/reg_CT_MRI.py` exists), then consider it as search space or weak supervision for
  an MRI pleura segmenter. Caution: thoracic CT↔MRI registration error vs pleura thickness,
  scan-interval effusion changes.
- **Registered vs original domain**: measure curves on native intensities via the
  inverse-warped superspel masks (already computed in `experiments/inv_mask_superpixel/`);
  robustness analysis, becomes mandatory before any texture-feature clinical claim.
- **Modernization swaps**: TotalSegmentator / nnU-Net effusion segmentation (removes
  interactive seeding), MedSAM-style prompting, learned registration (SynthMorph-class).
- **Mask v2** (lung-surface band) — superseded in priority by the CT-informed mask above.

## Phase 3.5 — cohort-level "superspel atlas" (linked-views pattern)

The linked-views pattern (scatter of embedded regions ↔ image gallery ↔ annotation panel,
all local) fits this project at cohort scale, once radiomics features exist:

- **Export superspels with embeddings**: one row per superspel (all 56 patients) with
  curve-shape / radiomics features and a 4D image crop (T×Z×Y×X) — e.g. as AnnData Zarr
  via a `meso-regions export-atlas` command.
- **Cohort browser**: scatter of all superspels coloured by histology / subclass / ECE call;
  click a cluster → see the underlying MRI regions. Useful for understanding BAPE false
  positives and for the radiomics second pass. Implementation options: a small Plotly
  Dash / Panel app built here (self-contained, no external dependencies), or an existing
  embedding browser if licensing/affiliation permits.
- **Annotation workflow**: the pleura / non-pleura expert slice review becomes an
  in-browser labelling step with exportable labels.

## Phase 4 — community & clinical usability

- [ ] CLI stable, versioned releases on PyPI
- [ ] Example notebook end-to-end on a public/synthetic case
- [ ] Contribution guide, issue templates
- [ ] GUI decision deferred (CLI-only for now); revisit 3D Slicer extension when there is a
      clinical pull for it
