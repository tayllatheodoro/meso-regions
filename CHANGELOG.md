# Changelog

All notable changes to Meso-Regions are documented here.
Versioning follows [SemVer](https://semver.org); 0.x versions may break APIs between minors.

## [0.8.1] — 2026-08-18

### Fixed
- `meso-regions run` no longer crashes when cohort metrics are undefined (single patient /
  single class); per-patient outputs are always written and the output path is printed.

### Verified
- Full pipeline smoke test with REAL data (patient 14): DICOM-lineage volumes →
  dilate → SLIC → ECE via the CLI; correct malignant call (78 ECE+ superspels).
  Found + fixed: published dilated masks are on the resampled 1.78 grid (225³) and do not
  match native-grid images — use fluid GT + the pipeline's dilate stage instead.

## [0.8.0] — 2026-08-18

### Added
- `meso-regions convert` — DICOM→NIfTI ingestion (`meso_regions/convert.py`): walks a tree
  of DICOM series folders and produces the `<patient>_<time>.nii.gz` layout the pipeline
  expects. dcm2niix primary, SimpleITK fallback; success judged by output presence.
- Claude Code skill (`.claude/skills/run-meso-regions/`) — agents can drive the full
  pipeline conversationally: convert → configure → dry-run → run → report.
- README "From DICOM" quickstart.

## [0.7.0] — 2026-08-18

### Added
- `meso_regions/experimental/dynamic_band.py` — EXPERIMENTAL enhancement-guided dynamic
  pleural band (lungs∪effusion surface, per-sector t270−t0 profiles, adaptive extent,
  anatomical organ exclusion). Expert-validated prototype on 2 patients; not used in
  published results. See ROADMAP Phase 2.

## [0.6.0] — 2026-08-17

### Added
- Gradio demo app (`demo/app.py`): browser UI over one example case — slice slider,
  region isolation dropdown linked to the curves plot, patient-level call. Example data
  lives in gitignored `demo/data/` (PREDICT-Meso governance) until a shareable case is
  confirmed; ready to publish as a Hugging Face Space then.

### Changed
- Roadmap: cohort atlas phrased implementation-neutral (Dash/Panel app or an existing
  embedding browser), not tied to any specific external tool

## [0.5.0] — 2026-08-17

### Added
- Report: linked interactive viewer — the slice viewer is now a client-side canvas
  compositing the embedded ECE label map, and the curves are clickable inline SVG.
  Clicking a region isolates its curve; clicking a curve isolates its region
  (Esc or re-click to reset). Mouse-wheel and slider slice navigation retained.
  Still a single self-contained file with no JS libraries and no data upload.

## [0.4.1] — 2026-08-17

### Changed
- Report: per-region colors now only on ECE-positive regions (supervoxels back to a single
  color); each rule-positive curve is drawn in its region's color and annotated with the
  region id, so curves can be matched to the overlays (curve row index == region label)
- Report: per-region layers rendered at higher opacity so hues match the curve colors

## [0.4.0] — 2026-08-17

### Added
- Report: per-region colors for label-map layers (supervoxels, ECE regions) and a new
  `--supervoxels` layer option
- Report: interactive slice slider (`--scroll-plane`, mouse-wheel support, capped at 72
  pre-rendered frames; `--no-scroll` to omit) — still a single self-contained file

## [0.3.0] — 2026-08-17

### Added
- `meso-regions report` — self-contained HTML patient report (`meso_regions/report.py`):
  axial/coronal/sagittal overlays of fluid / pleural-region / ECE masks, per-superspel
  signal-intensity curves classified in-place by the published ECE rule, mask volume table,
  patient-level ECE call, and research-use disclaimer. Single file, no external assets,
  no data upload.
- README: report usage and how to view outputs in 3D Slicer / ITK-SNAP

## [0.2.0] — 2026-08-17

### Added
- `meso-regions` CLI (`meso_regions/cli.py`): `stages`, `example-config`, and
  `run -c config.yaml [--dry-run]` — YAML-driven pipeline execution replacing the
  hard-coded paths in the per-stage scripts
- Console entry point and `pyyaml` dependency in `pyproject.toml`

## [0.1.0] — 2026-08-17

First installable release of the research code behind the ISBI 2026 paper.

### Added
- `pyproject.toml` — the project installs with `pip install -e .`
- README with pipeline overview, install steps, research-use-only disclaimer,
  PREDICT-Meso data access, and citations
- `docs/ROADMAP.md` — packaging/CLI, pleural mask v2, modernization candidates
- `.gitignore`, this changelog

### Changed
- Package renamed `src` → `meso_regions`; all internal imports rewritten

## [pre-0.1.0]

Research code as used for Theodoro et al., IEEE ISBI 2026 (commit `5706ad9` and earlier).
