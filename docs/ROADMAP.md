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

## Phase 2 — pleural mask v2

The current mask (effusion dilation minus fluid) has two known failure modes, observed by T.M.T.:
pleural lumps around the lungs that lie outside the band, and non-pleural tissue inside it.

- [ ] Lung-segmentation-guided band: union of (lung-surface rim) and (effusion rim), then refine —
      candidate lung masks from TotalSegmentator or the existing `lung_segmentation` experiments
- [ ] Quantify coverage change vs the published band (per-patient volume, overlap with expert
      ECE regions, effect on the 3-split classification metrics)
- [ ] Adaptive dilation radius by disease morphology (macro-nodular cases needed 3 voxels vs 2)

## Phase 3 — modernization candidates (field moved since ~2023)

Swap-in replacements to evaluate, one at a time, against the baseline:

- **Segmentation**: TotalSegmentator (lungs, +pleural effusion class where available),
  nnU-Net trained on the released 4D effusion ground truth (n=56) — could remove the
  interactive seeding step entirely
- **Interactive fallback**: MedSAM / SAM-Med3D-style promptable segmentation instead of
  IFT seed placement
- **Registration**: learning-based deformable registration (SynthMorph/VoxelMorph-class) vs
  ANTs SyN — faster, possibly better for large effusion deformation
- **Descriptor**: per-superspel radiomics feature trajectories (PyRadiomics) replacing
  mean-intensity curves — planned second-pass experiment for the journal paper
- **Classifier**: learned patient-level model over superspel features vs the rule — only with
  proper nested validation given n=56

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
